#include "models.h"

void llama_model_llama::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer()) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer()) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

void llama_model_llama::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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
        res->t_layer_inp[il] = inpL;

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

            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

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
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
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
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);

        cb(cur, "result_output", -1);
        // 保存 logits 结果
        res->t_logits = cur;
    }

    // 将最终节点添加到计算图中
    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_llama::graph<false>;
template struct llama_model_llama::graph<true>;
