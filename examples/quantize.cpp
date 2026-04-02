// sam3_quantize — Quantize SAM3 model weights from F32/F16 to Q4_0/Q4_1/Q8_0
//
// Usage: sam3_quantize <input.ggml> <output.ggml> <type>
//   types: q4_0, q4_1, q8_0

#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint32_t SAM3_MAGIC   = 0x73616D33;  // "sam3"
static constexpr uint32_t SAM2_MAGIC   = 0x73616D32;  // "sam2"
static constexpr int      SAM3_VERSION = 3;
static constexpr int      SAM2_VERSION = 1;

static bool sam3_quantize_model(const std::string & fname_inp,
                                const std::string & fname_out,
                                ggml_type qtype) {
    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // ── Read + write header (16 bytes) ──────────────────────────────────────
    uint32_t magic;
    int32_t  version, ftype, n_tensors;

    finp.read(reinterpret_cast<char *>(&magic),     4);
    finp.read(reinterpret_cast<char *>(&version),   4);
    finp.read(reinterpret_cast<char *>(&ftype),     4);
    finp.read(reinterpret_cast<char *>(&n_tensors), 4);

    bool is_sam2 = false;
    if (magic == SAM3_MAGIC) {
        if (version != SAM3_VERSION) {
            fprintf(stderr, "%s: unsupported SAM3 version %d (expected %d)\n",
                    __func__, version, SAM3_VERSION);
            return false;
        }
    } else if (magic == SAM2_MAGIC) {
        if (version != SAM2_VERSION) {
            fprintf(stderr, "%s: unsupported SAM2 version %d (expected %d)\n",
                    __func__, version, SAM2_VERSION);
            return false;
        }
        is_sam2 = true;
    } else {
        fprintf(stderr, "%s: unknown magic 0x%08x\n", __func__, magic);
        return false;
    }

    const int32_t ftype_out = static_cast<int32_t>(qtype);

    fout.write(reinterpret_cast<const char *>(&magic),     4);
    fout.write(reinterpret_cast<const char *>(&version),   4);
    fout.write(reinterpret_cast<const char *>(&ftype_out), 4);
    fout.write(reinterpret_cast<const char *>(&n_tensors), 4);

    fprintf(stderr, "%s: %s v%d, ftype %d -> %d (%s), %d tensors\n",
            __func__, is_sam2 ? "SAM2" : "SAM3", version, ftype,
            ftype_out, ggml_type_name(qtype), n_tensors);

    // ── Read + write hparams (copy through) ─────────────────────────────────
    // The number of hparam fields differs between SAM2 and SAM3.
    // SAM2: 50 int32 fields; SAM3: variable (depends on n_global_attn).
    // We copy them all byte-for-byte.
    auto copy_i32 = [&]() -> int32_t {
        int32_t v;
        finp.read(reinterpret_cast<char *>(&v), 4);
        fout.write(reinterpret_cast<const char *>(&v), 4);
        return v;
    };

    if (is_sam2) {
        // SAM2 standard header: 57 int32 fields total
        // Fields 1-2: img_size, backbone_type
        copy_i32();                           // img_size
        int32_t backbone_type = copy_i32();   // backbone_type (1=hiera, 2=repvit)
        // Fields 3-57: remaining standard SAM2 fields
        for (int i = 0; i < 55; ++i) copy_i32();
        // EdgeTAM extension: 19 additional fields when backbone_type == 2
        // (repvit_num_stages + stages[4] + channels[4] + se_ratio + has_perceiver +
        //  perceiver_depth + perceiver_dim + n_latents_1d + n_latents_2d + ff_mult +
        //  mem_attn_ca_type + ca_q_size + ca_k_size)
        if (backbone_type == 2) {
            for (int i = 0; i < 19; ++i) copy_i32();
        }
    } else {
        // SAM3 header
        copy_i32();  // img_size
        copy_i32();  // patch_size
        copy_i32();  // vit_embed_dim
        copy_i32();  // vit_depth
        copy_i32();  // vit_num_heads
        copy_i32();  // vit_mlp_ratio_x1000
        copy_i32();  // vit_window_size
        const int32_t n_global_attn = copy_i32();
        for (int i = 0; i < n_global_attn && i < 4; ++i) {
            copy_i32();  // global_attn_idx[i]
        }
        copy_i32();  // text_width
        copy_i32();  // text_heads
        copy_i32();  // text_layers
        copy_i32();  // text_ctx_len
        copy_i32();  // text_vocab_size
        copy_i32();  // text_out_dim
        copy_i32();  // neck_dim
        copy_i32();  // fenc_layers
        copy_i32();  // fenc_heads
        copy_i32();  // fenc_ffn_dim
        copy_i32();  // ddec_layers
        copy_i32();  // ddec_heads
        copy_i32();  // ddec_ffn_dim
        copy_i32();  // ddec_num_queries
        copy_i32();  // geom_layers
        copy_i32();  // n_presence_tokens
        copy_i32();  // n_geom_queries
        copy_i32();  // sam_embed_dim
        copy_i32();  // sam_dec_depth
        copy_i32();  // sam_n_multimask
        copy_i32();  // sam_iou_head_depth
        copy_i32();  // mem_out_dim
        copy_i32();  // mem_attn_layers
        copy_i32();  // num_maskmem
        copy_i32();  // max_obj_ptrs
        copy_i32();  // n_amb_experts
        copy_i32();  // visual_only
    }

    if (finp.fail()) {
        fprintf(stderr, "%s: failed to read hparams\n", __func__);
        return false;
    }

    // ── Process tensors ─────────────────────────────────────────────────────
    const int blk_size = ggml_blck_size(qtype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    int    n_quantized    = 0;
    int    n_total        = 0;

    std::vector<float>       data_f32;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<uint8_t>     work;
    std::vector<char>        data_raw;

    for (int t = 0; t < n_tensors; ++t) {
        int32_t n_dims, name_len, dtype;
        finp.read(reinterpret_cast<char *>(&n_dims),   4);
        finp.read(reinterpret_cast<char *>(&name_len), 4);
        finp.read(reinterpret_cast<char *>(&dtype),    4);
        if (finp.fail()) break;

        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            finp.read(reinterpret_cast<char *>(&ne[i]), 4);
        }

        std::string name(name_len, '\0');
        finp.read(&name[0], name_len);

        // Skip input padding to 32-byte alignment
        {
            size_t pos = finp.tellg();
            size_t pad = (32 - pos % 32) % 32;
            if (pad > 0) finp.seekg(pad, std::ios::cur);
        }

        int64_t n_el = 1;
        for (int i = 0; i < n_dims; ++i) n_el *= ne[i];

        // Determine input data size
        const ggml_type file_type = static_cast<ggml_type>(dtype);
        size_t inp_bytes;
        if (ggml_is_quantized(file_type)) {
            const int64_t n_rows = n_el / ne[0];
            inp_bytes = ggml_row_size(file_type, ne[0]) * n_rows;
        } else {
            const size_t elem_size = (file_type == GGML_TYPE_F16) ? 2 : 4;
            inp_bytes = n_el * elem_size;
        }

        // Decide whether to quantize:
        //  - must be 2D+ (skip biases, norms, scale factors)
        //  - ne[0] must be divisible by quantization block size
        //  - must not already be quantized
        //  - skip embeddings, tokens, positional encodings (registered as F32 in the model)
        auto name_contains = [&](const char * sub) {
            return name.find(sub) != std::string::npos;
        };
        // Match tensors registered as F32 in sam3_register_tensors (T2f/T3f/T4f).
        // These are embeddings, lookup tables, positional encodings, and special tokens
        // that must NOT be quantized.  Be specific to avoid catching weight matrices
        // like bbox_embed, mask_embed, or boxRPB_embed which ARE quantizable.
        const bool is_embedding =
            name_contains("token_embed")   || name_contains("pos_embed")
         || name_contains("query_embed")   || name_contains("label_embed")
         || name_contains("cls_embed")     || name_contains("point_embeddings")
         || name_contains("not_a_point_embed") || name_contains("no_mask_embed")
         || name_contains("no_mem_embed")  || name_contains("no_obj_embed")
         || name_contains("presence_token.weight")
         || name_contains("iou_token")     || name_contains("mask_tokens")
         || name_contains("obj_score_token")
         || name_contains("pe_gaussian")   || name_contains("freqs_cis")
         || name_contains("gamma")         || name_contains("tpos_enc")
         || name_contains("no_obj_ptr")    || name_contains("no_mem_pos_enc")
         || name_contains("trk_mask_ds")   || name_contains("latents");
        const bool quantize = (n_dims >= 2) &&
                              (ne[0] % blk_size == 0) &&
                              !ggml_is_quantized(file_type) &&
                              !is_embedding;

        if (quantize) {
            // Read and convert to F32
            if (file_type == GGML_TYPE_F16) {
                data_f16.resize(n_el);
                finp.read(reinterpret_cast<char *>(data_f16.data()), inp_bytes);
                data_f32.resize(n_el);
                for (int64_t i = 0; i < n_el; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(n_el);
                finp.read(reinterpret_cast<char *>(data_f32.data()), inp_bytes);
            }

            // Quantize
            const int64_t n_rows     = n_el / ne[0];
            const size_t  out_row_sz = ggml_row_size(qtype, ne[0]);
            work.resize(n_rows * out_row_sz);

            const size_t cur_size = ggml_quantize_chunk(
                qtype, data_f32.data(), work.data(), 0, n_rows, ne[0], nullptr);

            // Write tensor header with quantized dtype
            const int32_t dtype_out = static_cast<int32_t>(qtype);
            fout.write(reinterpret_cast<const char *>(&n_dims),    4);
            fout.write(reinterpret_cast<const char *>(&name_len),  4);
            fout.write(reinterpret_cast<const char *>(&dtype_out), 4);
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<const char *>(&ne[i]), 4);
            }
            fout.write(name.data(), name_len);

            // Pad to 32-byte alignment
            {
                size_t pos = fout.tellp();
                size_t pad = (32 - pos % 32) % 32;
                const char zero = 0;
                for (size_t i = 0; i < pad; ++i) fout.write(&zero, 1);
            }

            fout.write(reinterpret_cast<const char *>(work.data()), cur_size);

            total_size_new += cur_size;
            n_quantized++;

            printf("%64s - [%5d, %5d, %5d, %5d] %6s -> %6s  %8.2f MB -> %8.2f MB\n",
                   name.c_str(), ne[0], ne[1], ne[2], ne[3],
                   ggml_type_name(file_type), ggml_type_name(qtype),
                   inp_bytes / (1024.0 * 1024.0),
                   cur_size  / (1024.0 * 1024.0));
        } else {
            // Copy tensor as-is
            data_raw.resize(inp_bytes);
            finp.read(data_raw.data(), inp_bytes);

            fout.write(reinterpret_cast<const char *>(&n_dims),   4);
            fout.write(reinterpret_cast<const char *>(&name_len), 4);
            fout.write(reinterpret_cast<const char *>(&dtype),    4);
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<const char *>(&ne[i]), 4);
            }
            fout.write(name.data(), name_len);

            // Pad to 32-byte alignment
            {
                size_t pos = fout.tellp();
                size_t pad = (32 - pos % 32) % 32;
                const char zero = 0;
                for (size_t i = 0; i < pad; ++i) fout.write(&zero, 1);
            }

            fout.write(data_raw.data(), inp_bytes);

            total_size_new += inp_bytes;

            printf("%64s - [%5d, %5d, %5d, %5d] %6s  (kept)  %8.2f MB\n",
                   name.c_str(), ne[0], ne[1], ne[2], ne[3],
                   ggml_type_name(file_type),
                   inp_bytes / (1024.0 * 1024.0));
        }

        total_size_org += n_el * sizeof(float);
        n_total++;
    }

    // ── Copy remaining data (embedded tokenizer block) verbatim ───────────
    {
        char buf[4096];
        while (finp.read(buf, sizeof(buf))) {
            fout.write(buf, finp.gcount());
        }
        if (finp.gcount() > 0) {
            fout.write(buf, finp.gcount());
        }
    }

    printf("\n");
    printf("%s: quantized %d / %d tensors\n", __func__, n_quantized, n_total);
    printf("%s: original size  = %8.2f MB (F32 equivalent)\n",
           __func__, total_size_org / (1024.0 * 1024.0));
    printf("%s: quantized size = %8.2f MB\n",
           __func__, total_size_new / (1024.0 * 1024.0));
    printf("%s: compression    = %.2fx\n",
           __func__, (double)total_size_org / total_size_new);

    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.ggml model-quant.ggml type\n", argv[0]);
        fprintf(stderr, "  supported types: q4_0, q4_1, q8_0\n");
        return 1;
    }

    // Init ggml (needed for fp16 lookup tables)
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    ggml_type qtype = GGML_TYPE_COUNT;
    if      (strcmp(argv[3], "q4_0") == 0) qtype = GGML_TYPE_Q4_0;
    else if (strcmp(argv[3], "q4_1") == 0) qtype = GGML_TYPE_Q4_1;
    else if (strcmp(argv[3], "q8_0") == 0) qtype = GGML_TYPE_Q8_0;
    else {
        fprintf(stderr, "%s: unknown quantization type '%s'\n", argv[0], argv[3]);
        fprintf(stderr, "  supported types: q4_0, q4_1, q8_0\n");
        return 1;
    }

    fprintf(stderr, "%s: quantizing '%s' -> '%s' (%s)\n",
            __func__, fname_inp.c_str(), fname_out.c_str(), ggml_type_name(qtype));

    const int64_t t_start = ggml_time_us();

    if (!sam3_quantize_model(fname_inp, fname_out, qtype)) {
        fprintf(stderr, "%s: failed to quantize model\n", __func__);
        return 1;
    }

    const double t_elapsed = (ggml_time_us() - t_start) / 1e6;
    printf("%s: quantize time = %.2f s\n", __func__, t_elapsed);

    return 0;
}
