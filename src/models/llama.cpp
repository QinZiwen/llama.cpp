#include "models.h"

template <bool embed>
llm_build_llama<embed>::llm_build_llama(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // 获取注意力头的维度大小
    const int64_t n_embd_head = hparams.n_embd_head_v();

    // 断言：K和V的头维度必须相等，且等于旋转位置编码的维度
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;      // 当前层的输出张量
    ggml_tensor * inpL;     // 当前层的输入张量（残差连接输入）

    // 构建输入嵌入层：将token ID转换为嵌入向量，或直接使用输入嵌入
    inpL = build_inp_embd(model.tok_embd);

    // 构建输入位置编码张量
    ggml_tensor * inp_pos = build_inp_pos();

    // 根据是否为嵌入模式选择注意力输入类型：
    // embed=true: 无KV缓存（用于编码器或仅嵌入任务）
    // embed=false: 使用KV缓存（用于解码器生成）
    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        // 构建无缓存的注意力输入（不存储KV）
        inp_attn = build_attn_inp_no_cache();
    } else {
        // 构建带KV缓存的注意力输入
        inp_attn = build_attn_inp_kv();
    }

    // 计算注意力缩放因子：如果未指定，则使用标准的 1/sqrt(d_k)
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // 构建输出ID索引张量：用于在最后一层只提取需要输出的token
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // 遍历所有Transformer层
    for (int il = 0; il < n_layer; ++il) {
        // 保存当前层的输入，用于残差连接
        ggml_tensor * inpSA = inpL;

        // --- 注意力模块前归一化 (RMSNorm) ---
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- 自注意力机制 ---
        {
            // 获取RoPE频率因子（Llama3可能需要，Llama2等可能返回nullptr）
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // 计算查询向量 Q = X @ W_q
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            cb(Qcur, "Qcur", il);
            // 如果存在偏置，加上偏置
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            // 计算键向量 K = X @ W_k
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            // 计算值向量 V = X @ W_v
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            // 重塑张量以分离多头维度: [n_tokens, n_embd] -> [n_embd_head, n_head, n_tokens]
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // 对 Q 应用旋转位置编码 (RoPE)
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // 对 K 应用旋转位置编码 (RoPE)
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // 如果启用了QK归一化（如Llama4），对Q和K进行RMSNorm
            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }

            // 构建注意力输出: Attention(Q, K, V) @ W_o
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            
            // 如果存在输出缩放因子，应用缩放
            if (model.layers[il].wo_s) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_s);
            }
            cb(cur, "attn_out", il);
        }

        // 如果是最后一层且存在输出ID过滤，则只保留需要输出的token
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // 残差连接: H = H + Attention(H)
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- 前馈神经网络 (FFN) ---
        // 判断是否是MoE模型：如果ffn_gate_inp为空，则为标准 dense FFN
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // 标准 FFN 分支

            // FFN前归一化 (RMSNorm)
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // 构建标准 FFN: Down(Silu(Gate(X)) * Up(X))
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE (Mixture of Experts) 分支

            // FFN前归一化 (RMSNorm)
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // 构建 MoE FFN
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,      // 专家门控输入权重
                    model.layers[il].ffn_up_exps,       // 专家Up投影权重
                    model.layers[il].ffn_gate_exps,     // 专家Gate投影权重
                    model.layers[il].ffn_down_exps,     // 专家Down投影权重
                    nullptr,                            // 专家概率偏置（可选）
                    n_expert, n_expert_used,            // 专家总数和激活专家数
                    LLM_FFN_SILU, true,                 // 激活函数和是否归一化权重
                    hparams.expert_weights_scale,       // 专家权重缩放
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, // 门控函数类型
                    il,
                    nullptr, nullptr,                   // 其他可选输入
                    model.layers[il].ffn_up_exps_s,     // LoRA缩放
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }

        // 残差连接: H = H + FFN(H)
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        // 应用自定义向量控制 (Control Vector)
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // 更新下一层的输入
        inpL = cur;
    }

    // 最终层输出
    cur = inpL;

    // 最终归一化 (RMSNorm)
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    // 保存嵌入结果（用于提取嵌入向量）
    res->t_embd = cur;

    // 如果不是仅嵌入模式，则计算 logits
    if constexpr (!embed) {
        // 语言模型头: Logits = Embeddings @ Output_Weights
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        // 保存 logits 结果
        res->t_logits = cur;
    }

    // 将最终节点添加到计算图中
    ggml_build_forward_expand(gf, cur);
}

template struct llm_build_llama<false>;
template struct llm_build_llama<true>;
