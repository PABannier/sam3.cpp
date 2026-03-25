#include "sam3.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr uint32_t SAM3_MAGIC = 0x73616D33;  // "sam3"
static constexpr int SAM3_VERSION = 1;

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal data types — hyperparameters
// ═══════════════════════════════════════════════════════════════════════════════

struct sam3_hparams {
    int32_t img_size = 1008;
    int32_t patch_size = 14;
    int32_t vit_embed_dim = 1024;
    int32_t vit_depth = 32;
    int32_t vit_num_heads = 16;
    int32_t vit_mlp_dim = 4736;  // 1024 * 4.625
    int32_t vit_window_size = 24;
    int32_t n_global_attn = 4;
    int32_t global_attn_idx[4] = {7, 15, 23, 31};

    int32_t text_width = 1024;
    int32_t text_heads = 16;
    int32_t text_layers = 24;
    int32_t text_ctx_len = 32;
    int32_t text_vocab_size = 49408;
    int32_t text_out_dim = 256;

    int32_t neck_dim = 256;

    int32_t fenc_layers = 6;
    int32_t fenc_heads = 8;
    int32_t fenc_ffn_dim = 2048;

    int32_t ddec_layers = 6;
    int32_t ddec_heads = 8;
    int32_t ddec_ffn_dim = 2048;
    int32_t ddec_num_queries = 200;

    int32_t geom_layers = 3;
    int32_t n_presence_tokens = 1;
    int32_t n_geom_queries = 4;

    int32_t sam_embed_dim = 256;
    int32_t sam_dec_depth = 2;
    int32_t sam_n_multimask = 3;
    int32_t sam_iou_head_depth = 3;

    int32_t mem_out_dim = 64;
    int32_t mem_attn_layers = 4;
    int32_t num_maskmem = 7;
    int32_t max_obj_ptrs = 16;

    int32_t n_amb_experts = 2;

    // derived helpers
    int32_t n_img_embd() const { return img_size / patch_size; }            // 72
    int32_t n_img_tokens() const { return n_img_embd() * n_img_embd(); }    // 5184
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }  // 64

    bool is_global_attn(int layer) const {
        for (int i = 0; i < n_global_attn; ++i) {
            if (global_attn_idx[i] == layer) return true;
        }
        return false;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal data types — layer weight structs
// ═══════════════════════════════════════════════════════════════════════════════

// ── ViT backbone ─────────────────────────────────────────────────────────────

struct sam3_vit_block {
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    struct ggml_tensor* qkv_w = nullptr;
    struct ggml_tensor* qkv_b = nullptr;
    struct ggml_tensor* proj_w = nullptr;
    struct ggml_tensor* proj_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
    struct ggml_tensor* freqs_cis = nullptr;  // [N, 32, 2] RoPE (N=576 window, 5184 global)
};

struct sam3_vit {
    struct ggml_tensor* patch_embed_w = nullptr;  // [patch, patch, 3, embed] (ggml conv kernel)
    struct ggml_tensor* pos_embed = nullptr;      // [embed, 24, 24, 1] (pretrained res, tiled at runtime)
    struct ggml_tensor* ln_pre_w = nullptr;
    struct ggml_tensor* ln_pre_b = nullptr;
    std::vector<sam3_vit_block> blocks;
};

// ── Neck (SimpleFPN) ─────────────────────────────────────────────────────────

struct sam3_neck_scale {
    struct ggml_tensor* deconv1_w = nullptr;
    struct ggml_tensor* deconv1_b = nullptr;
    struct ggml_tensor* deconv2_w = nullptr;  // only for 4x scale
    struct ggml_tensor* deconv2_b = nullptr;
    struct ggml_tensor* conv1x1_w = nullptr;
    struct ggml_tensor* conv1x1_b = nullptr;
    struct ggml_tensor* conv3x3_w = nullptr;
    struct ggml_tensor* conv3x3_b = nullptr;
};

struct sam3_neck {
    sam3_neck_scale scales[4];
    struct ggml_tensor* norms_w[4] = {};
    struct ggml_tensor* norms_b[4] = {};
};

// ── Text encoder ─────────────────────────────────────────────────────────────

struct sam3_text_block {
    struct ggml_tensor* attn_in_proj_w = nullptr;
    struct ggml_tensor* attn_in_proj_b = nullptr;
    struct ggml_tensor* attn_out_proj_w = nullptr;
    struct ggml_tensor* attn_out_proj_b = nullptr;
    struct ggml_tensor* ln1_w = nullptr;
    struct ggml_tensor* ln1_b = nullptr;
    struct ggml_tensor* ln2_w = nullptr;
    struct ggml_tensor* ln2_b = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
    struct ggml_tensor* ls1 = nullptr;  // LayerScale (may be null)
    struct ggml_tensor* ls2 = nullptr;
};

struct sam3_text_encoder {
    struct ggml_tensor* token_embed_w = nullptr;  // [vocab, width]
    struct ggml_tensor* pos_embed = nullptr;      // [ctx_len, width]
    struct ggml_tensor* ln_final_w = nullptr;
    struct ggml_tensor* ln_final_b = nullptr;
    struct ggml_tensor* resizer_w = nullptr;  // [out_dim, width]
    struct ggml_tensor* resizer_b = nullptr;
    // Note: text_projection ([width, proj_dim]) exists in the checkpoint but is
    // intentionally not loaded. In SAM3, VETextEncoder discards the pooled output
    // that text_projection operates on — only the full token sequence (through
    // resizer) is used for downstream fusion/decoding.
    std::vector<sam3_text_block> blocks;
};

// ── Fusion encoder ───────────────────────────────────────────────────────────

struct sam3_fenc_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w = nullptr;
    struct ggml_tensor* sa_in_proj_b = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    // cross-attention to prompt tokens
    struct ggml_tensor* ca_q_w = nullptr;
    struct ggml_tensor* ca_q_b = nullptr;
    struct ggml_tensor* ca_kv_w = nullptr;
    struct ggml_tensor* ca_kv_b = nullptr;
    struct ggml_tensor* ca_out_w = nullptr;
    struct ggml_tensor* ca_out_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm3_w = nullptr;
    struct ggml_tensor* norm3_b = nullptr;
};

struct sam3_fusion_encoder {
    std::vector<sam3_fenc_layer> layers;
};

// ── DETR decoder ─────────────────────────────────────────────────────────────

struct sam3_ddec_layer {
    // self-attention
    struct ggml_tensor* sa_in_proj_w = nullptr;
    struct ggml_tensor* sa_in_proj_b = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    // cross-attention to image
    struct ggml_tensor* ca_q_w = nullptr;
    struct ggml_tensor* ca_q_b = nullptr;
    struct ggml_tensor* ca_kv_w = nullptr;
    struct ggml_tensor* ca_kv_b = nullptr;
    struct ggml_tensor* ca_out_w = nullptr;
    struct ggml_tensor* ca_out_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    // cross-attention to text
    struct ggml_tensor* ca_text_q_w = nullptr;
    struct ggml_tensor* ca_text_q_b = nullptr;
    struct ggml_tensor* ca_text_kv_w = nullptr;
    struct ggml_tensor* ca_text_kv_b = nullptr;
    struct ggml_tensor* ca_text_out_w = nullptr;
    struct ggml_tensor* ca_text_out_b = nullptr;
    struct ggml_tensor* norm3_w = nullptr;
    struct ggml_tensor* norm3_b = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm4_w = nullptr;
    struct ggml_tensor* norm4_b = nullptr;
    // box refinement MLP (3 layers)
    struct ggml_tensor* bbox_w[3] = {};
    struct ggml_tensor* bbox_b[3] = {};
};

struct sam3_detr_decoder {
    struct ggml_tensor* query_embed = nullptr;     // [num_queries, 512]
    struct ggml_tensor* presence_token = nullptr;  // [1, 256]
    // DotProductScoring MLP
    struct ggml_tensor* score_mlp_w[2] = {};
    struct ggml_tensor* score_mlp_b[2] = {};
    struct ggml_tensor* score_ln_w = nullptr;
    struct ggml_tensor* score_ln_b = nullptr;
    // Presence head
    struct ggml_tensor* presence_head_w[2] = {};
    struct ggml_tensor* presence_head_b[2] = {};
    std::vector<sam3_ddec_layer> layers;
};

// ── Geometry / exemplar encoder ──────────────────────────────────────────────

struct sam3_geom_layer {
    struct ggml_tensor* sa_in_proj_w = nullptr;
    struct ggml_tensor* sa_in_proj_b = nullptr;
    struct ggml_tensor* sa_out_proj_w = nullptr;
    struct ggml_tensor* sa_out_proj_b = nullptr;
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    struct ggml_tensor* ca_q_w = nullptr;
    struct ggml_tensor* ca_q_b = nullptr;
    struct ggml_tensor* ca_kv_w = nullptr;
    struct ggml_tensor* ca_kv_b = nullptr;
    struct ggml_tensor* ca_out_w = nullptr;
    struct ggml_tensor* ca_out_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm3_w = nullptr;
    struct ggml_tensor* norm3_b = nullptr;
};

struct sam3_geom_encoder {
    struct ggml_tensor* point_proj_w = nullptr;
    struct ggml_tensor* point_proj_b = nullptr;
    struct ggml_tensor* box_proj_w = nullptr;
    struct ggml_tensor* box_proj_b = nullptr;
    struct ggml_tensor* type_embed = nullptr;
    struct ggml_tensor* cls_token = nullptr;
    struct ggml_tensor* post_proj_w = nullptr;
    struct ggml_tensor* post_proj_b = nullptr;
    std::vector<sam3_geom_layer> layers;
};

// ── Segmentation head (MaskFormer) ───────────────────────────────────────────

struct sam3_seg_head {
    struct ggml_tensor* up_conv_w[3] = {};
    struct ggml_tensor* up_conv_b[3] = {};
    struct ggml_tensor* up_norm_w[3] = {};
    struct ggml_tensor* up_norm_b[3] = {};
    struct ggml_tensor* ca_prompt_q_w = nullptr;
    struct ggml_tensor* ca_prompt_q_b = nullptr;
    struct ggml_tensor* ca_prompt_kv_w = nullptr;
    struct ggml_tensor* ca_prompt_kv_b = nullptr;
    struct ggml_tensor* ca_prompt_out_w = nullptr;
    struct ggml_tensor* ca_prompt_out_b = nullptr;
    struct ggml_tensor* mask_embed_w = nullptr;
    struct ggml_tensor* mask_embed_b = nullptr;
};

// ── SAM prompt encoder (tracker path) ────────────────────────────────────────

struct sam3_sam_prompt_enc {
    struct ggml_tensor* pe_gaussian = nullptr;        // [2, 128]
    struct ggml_tensor* point_embed[4] = {};          // neg, pos, box_tl, box_br
    struct ggml_tensor* not_a_point_embed = nullptr;  // [256]
    struct ggml_tensor* no_mask_embed = nullptr;      // [256]
    struct ggml_tensor* mask_ds_conv_w[3] = {};
    struct ggml_tensor* mask_ds_conv_b[3] = {};
    struct ggml_tensor* mask_ds_norm_w[2] = {};
    struct ggml_tensor* mask_ds_norm_b[2] = {};
};

// ── SAM mask decoder (tracker path) ──────────────────────────────────────────

struct sam3_sam_attn {
    struct ggml_tensor* q_w = nullptr;
    struct ggml_tensor* q_b = nullptr;
    struct ggml_tensor* k_w = nullptr;
    struct ggml_tensor* k_b = nullptr;
    struct ggml_tensor* v_w = nullptr;
    struct ggml_tensor* v_b = nullptr;
    struct ggml_tensor* out_w = nullptr;
    struct ggml_tensor* out_b = nullptr;
};

struct sam3_twoway_block {
    sam3_sam_attn self_attn;
    sam3_sam_attn ca_tok2img;
    sam3_sam_attn ca_img2tok;
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    struct ggml_tensor* norm3_w = nullptr;
    struct ggml_tensor* norm3_b = nullptr;
    struct ggml_tensor* norm4_w = nullptr;
    struct ggml_tensor* norm4_b = nullptr;
    struct ggml_tensor* mlp_fc1_w = nullptr;
    struct ggml_tensor* mlp_fc1_b = nullptr;
    struct ggml_tensor* mlp_fc2_w = nullptr;
    struct ggml_tensor* mlp_fc2_b = nullptr;
};

struct sam3_sam_mask_dec {
    struct ggml_tensor* iou_token = nullptr;        // [1, 256]
    struct ggml_tensor* mask_tokens = nullptr;      // [4, 256]
    struct ggml_tensor* obj_score_token = nullptr;  // [1, 256]

    std::vector<sam3_twoway_block> twoway_blocks;  // [2]

    sam3_sam_attn final_attn;
    struct ggml_tensor* final_norm_w = nullptr;
    struct ggml_tensor* final_norm_b = nullptr;

    // upscaling
    struct ggml_tensor* up1_w = nullptr;
    struct ggml_tensor* up1_b = nullptr;
    struct ggml_tensor* up1_norm_w = nullptr;
    struct ggml_tensor* up1_norm_b = nullptr;
    struct ggml_tensor* up2_w = nullptr;
    struct ggml_tensor* up2_b = nullptr;

    // high-res feature convolutions
    struct ggml_tensor* conv_s0_w = nullptr;
    struct ggml_tensor* conv_s0_b = nullptr;
    struct ggml_tensor* conv_s1_w = nullptr;
    struct ggml_tensor* conv_s1_b = nullptr;

    // hypernetwork MLPs: 4 masks × 3 layers
    struct ggml_tensor* hyper_w[4][3] = {};
    struct ggml_tensor* hyper_b[4][3] = {};

    // IoU prediction head (3 layers)
    struct ggml_tensor* iou_head_w[3] = {};
    struct ggml_tensor* iou_head_b[3] = {};

    // object score head (3 layers)
    struct ggml_tensor* obj_head_w[3] = {};
    struct ggml_tensor* obj_head_b[3] = {};
};

// ── Memory encoder ───────────────────────────────────────────────────────────

struct sam3_mem_enc {
    // mask downsampler (4 conv stages + final 1x1)
    struct ggml_tensor* ds_conv_w[5] = {};
    struct ggml_tensor* ds_conv_b[5] = {};
    struct ggml_tensor* ds_norm_w[4] = {};
    struct ggml_tensor* ds_norm_b[4] = {};
    // pixel feature projection
    struct ggml_tensor* pix_proj_w = nullptr;
    struct ggml_tensor* pix_proj_b = nullptr;
    // fuser (2 CXBlock layers)
    struct ggml_tensor* fuser_dw_w[2] = {};
    struct ggml_tensor* fuser_dw_b[2] = {};
    struct ggml_tensor* fuser_norm_w[2] = {};
    struct ggml_tensor* fuser_norm_b[2] = {};
    struct ggml_tensor* fuser_fc1_w[2] = {};
    struct ggml_tensor* fuser_fc1_b[2] = {};
    struct ggml_tensor* fuser_fc2_w[2] = {};
    struct ggml_tensor* fuser_fc2_b[2] = {};
    struct ggml_tensor* fuser_gamma[2] = {};
    // output projection
    struct ggml_tensor* out_proj_w = nullptr;
    struct ggml_tensor* out_proj_b = nullptr;
    // temporal pos encodings
    struct ggml_tensor* tpos[7] = {};
};

// ── Memory attention (tracker transformer) ───────────────────────────────────

struct sam3_mem_attn_layer {
    // self-attention (RoPE, 1 head, 256-dim)
    struct ggml_tensor* sa_q_w = nullptr;
    struct ggml_tensor* sa_q_b = nullptr;
    struct ggml_tensor* sa_k_w = nullptr;
    struct ggml_tensor* sa_k_b = nullptr;
    struct ggml_tensor* sa_v_w = nullptr;
    struct ggml_tensor* sa_v_b = nullptr;
    struct ggml_tensor* sa_out_w = nullptr;
    struct ggml_tensor* sa_out_b = nullptr;
    struct ggml_tensor* norm1_w = nullptr;
    struct ggml_tensor* norm1_b = nullptr;
    // cross-attention (RoPE, kv_dim=64)
    struct ggml_tensor* ca_q_w = nullptr;
    struct ggml_tensor* ca_q_b = nullptr;
    struct ggml_tensor* ca_k_w = nullptr;  // [256, 64]
    struct ggml_tensor* ca_k_b = nullptr;
    struct ggml_tensor* ca_v_w = nullptr;  // [256, 64]
    struct ggml_tensor* ca_v_b = nullptr;
    struct ggml_tensor* ca_out_w = nullptr;
    struct ggml_tensor* ca_out_b = nullptr;
    struct ggml_tensor* norm2_w = nullptr;
    struct ggml_tensor* norm2_b = nullptr;
    // FFN
    struct ggml_tensor* ffn_fc1_w = nullptr;
    struct ggml_tensor* ffn_fc1_b = nullptr;
    struct ggml_tensor* ffn_fc2_w = nullptr;
    struct ggml_tensor* ffn_fc2_b = nullptr;
    struct ggml_tensor* norm3_w = nullptr;
    struct ggml_tensor* norm3_b = nullptr;
};

struct sam3_mem_attn {
    std::vector<sam3_mem_attn_layer> layers;
};

// ── BPE tokenizer ────────────────────────────────────────────────────────────

struct sam3_bpe_tokenizer {
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> merge_ranks;       // "a\x1fb" → rank
    std::unordered_map<uint8_t, std::string> byte_encoder;  // byte → unicode UTF-8
    std::unordered_map<std::string, std::string> cache;
    int sot_token = 49406;
    int eot_token = 49407;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Top-level opaque types (defined here, forward-declared in sam3.h)
// ═══════════════════════════════════════════════════════════════════════════════

struct sam3_model {
    sam3_hparams hparams;
    ggml_type weight_type = GGML_TYPE_F16;

    sam3_vit vit;
    sam3_neck neck_det;
    sam3_neck neck_trk;
    sam3_text_encoder text_enc;
    sam3_fusion_encoder fenc;
    sam3_detr_decoder ddec;
    sam3_geom_encoder geom_enc;
    sam3_seg_head seg_head;

    sam3_sam_prompt_enc sam_pe;
    sam3_sam_mask_dec sam_dec;
    sam3_mem_enc mem_enc;
    sam3_mem_attn mem_attn;

    // object pointer projection
    struct ggml_tensor* obj_ptr_proj_w[3] = {};
    struct ggml_tensor* obj_ptr_proj_b[3] = {};
    struct ggml_tensor* no_obj_ptr = nullptr;
    struct ggml_tensor* obj_ptr_tpos_w = nullptr;
    struct ggml_tensor* obj_ptr_tpos_b = nullptr;

    // precomputed RoPE frequencies
    struct ggml_tensor* rope_freqs = nullptr;  // [n_img_tokens, head_dim]

    // ggml backend
    struct ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // tensor lookup
    std::map<std::string, struct ggml_tensor*> tensors;

    // tokenizer
    sam3_bpe_tokenizer tokenizer;
};

struct sam3_state {
    // cached backbone outputs
    struct ggml_tensor* vit_output = nullptr;  // [1, embed, H, W]
    struct ggml_tensor* neck_det[4] = {};      // FPN levels (det path)
    struct ggml_tensor* neck_trk[4] = {};      // FPN levels (trk path)
    struct ggml_tensor* neck_det_pe[4] = {};   // sinusoidal PE
    struct ggml_tensor* neck_trk_pe[4] = {};

    int orig_width = 0;
    int orig_height = 0;
    int n_threads = 4;

    struct ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct ggml_gallocr* galloc = nullptr;

    // PE buffer: holds sinusoidal PE tensors for neck outputs
    struct ggml_context* pe_ctx = nullptr;
    ggml_backend_buffer_t pe_buf = nullptr;

    // Cached SAM prompt encoder embeddings (read from GPU once, reused per PVS call)
    bool pe_cache_valid = false;
    std::vector<float> pe_gauss_cache;       // [2 * num_pos_feats]
    float point_emb_cache[4][256] = {};
    float not_a_point_cache[256] = {};
    float no_mask_emb_cache[256] = {};
    std::vector<float> dense_pe_cache;       // [D * H * H] — positional encoding grid
    std::vector<float> dense_nomask_cache;   // [D * H * H] — no-mask embedding tiled
};

// ── Video tracker state ──────────────────────────────────────────────────────

struct sam3_masklet {
    int instance_id = -1;
    int first_frame = -1;
    int last_seen = -1;
    float last_score = 0.0f;
    bool confirmed = false;
    int mds_sum = 0;

    // last predicted mask logits (owned by tracker ctx)
    struct ggml_tensor* mask_logits = nullptr;  // [1, 1, 288, 288]
    struct ggml_tensor* obj_ptr = nullptr;      // [1, 256]
};

struct sam3_memory_slot {
    struct ggml_tensor* spatial_feats = nullptr;  // [64, 72, 72]
    struct ggml_tensor* spatial_pe = nullptr;     // [64, 72, 72]
    int frame_index = -1;
    bool is_cond_frame = false;
};

struct sam3_tracker {
    sam3_video_params params;
    int frame_index = 0;
    int next_inst_id = 1;

    std::vector<sam3_masklet> masklets;
    std::vector<sam3_masklet> pending;

    std::map<int, std::vector<sam3_memory_slot>> mem_banks;
    std::map<int, std::vector<std::pair<int, struct ggml_tensor*>>> ptr_banks;

    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // Per-tensor backend buffers allocated by sam3_encode_memory / sam3_store_obj_ptr.
    // Tracked here so they can be freed on tracker reset.
    std::vector<ggml_backend_buffer_t> owned_buffers;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal helper declarations
// ═══════════════════════════════════════════════════════════════════════════════

// graph execution
static void sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph, int n_threads);

// ggml building blocks
static struct ggml_tensor* sam3_layer_norm(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* w,
                                           struct ggml_tensor* b);

static struct ggml_tensor* sam3_layer_norm_2d(struct ggml_context* ctx,
                                              struct ggml_tensor* x,
                                              struct ggml_tensor* w,
                                              struct ggml_tensor* b);

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal helper implementations
// ═══════════════════════════════════════════════════════════════════════════════

static void sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph, int n_threads) {
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, graph);
}

static void sam3_name_tensorf(struct ggml_tensor * t, const char * fmt, int index) {
    if (!t) {
        return;
    }
    char name[64];
    snprintf(name, sizeof(name), fmt, index);
    ggml_set_name(t, name);
}

static struct ggml_tensor* sam3_layer_norm(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* w,
                                           struct ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static struct ggml_tensor* sam3_layer_norm_2d(struct ggml_context* ctx,
                                              struct ggml_tensor* x,
                                              struct ggml_tensor* w,
                                              struct ggml_tensor* b) {
    // x is [C, H, W, B] in ggml layout — norm over C dimension (dim 0)
    x = ggml_norm(ctx, x, 1e-6f);
    // w, b are [C, 1, 1] — broadcast multiply/add
    x = ggml_mul(ctx, x, w);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static struct ggml_tensor* sam3_conv_transpose_weight(struct ggml_context * ctx,
                                                      struct ggml_tensor * w) {
    return w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BPE Tokenizer — CLIP-style byte-level BPE
// ═══════════════════════════════════════════════════════════════════════════════

// ── UTF-8 helpers ────────────────────────────────────────────────────────────

static int sam3_utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;  // continuation (shouldn't start here)
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static std::string sam3_codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Check if position i in s starts a Unicode letter.
// Handles ASCII letters + treats any multibyte UTF-8 start byte as a letter.
// This is a reasonable approximation without ICU.
static bool sam3_is_letter(const std::string& s, size_t i) {
    uint8_t c = (uint8_t)s[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c >= 0xC0) return true;  // multibyte UTF-8 → treat as letter
    return false;
}

// ── Byte-to-unicode mapping (CLIP / GPT-2 style) ────────────────────────────

// Maps each byte 0-255 to a unique unicode character (as UTF-8 string).
// Printable bytes map to themselves; non-printable bytes map to U+0100..U+0143.
static void sam3_init_byte_encoder(std::unordered_map<uint8_t, std::string>& enc) {
    // Collect printable byte values
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    // Corresponding codepoints (printable → identity)
    std::vector<int> cs(bs.begin(), bs.end());

    // Non-printable bytes get codepoints starting at 256
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    enc.clear();
    for (size_t i = 0; i < bs.size(); ++i) {
        enc[(uint8_t)bs[i]] = sam3_codepoint_to_utf8(cs[i]);
    }
}

// ── Merge key helper ─────────────────────────────────────────────────────────

// Unit separator (0x1F) cannot appear in byte-encoded BPE tokens.
static inline std::string sam3_merge_key(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + 1 + b.size());
    k += a;
    k += '\x1f';
    k += b;
    return k;
}

// ── Minimal JSON parser for vocab.json ───────────────────────────────────────

// Parses a flat { "string": int, ... } JSON object.
static bool sam3_parse_vocab_json(const std::string& path,
                                  std::unordered_map<std::string, int>& encoder) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    size_t pos = 0;
    // Skip to '{'
    while (pos < content.size() && content[pos] != '{') pos++;
    if (pos >= content.size()) return false;
    pos++;

    while (pos < content.size()) {
        // Skip whitespace and commas
        while (pos < content.size() &&
               (content[pos] == ' ' || content[pos] == '\n' ||
                content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
            pos++;

        if (pos >= content.size() || content[pos] == '}') break;

        // Expect '"'
        if (content[pos] != '"') return false;
        pos++;

        // Read key (handle escape sequences)
        std::string key;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\') {
                pos++;
                if (pos >= content.size()) return false;
                switch (content[pos]) {
                    case '"':
                        key += '"';
                        break;
                    case '\\':
                        key += '\\';
                        break;
                    case '/':
                        key += '/';
                        break;
                    case 'n':
                        key += '\n';
                        break;
                    case 'r':
                        key += '\r';
                        break;
                    case 't':
                        key += '\t';
                        break;
                    case 'u': {
                        // Parse 4-hex-digit unicode escape
                        if (pos + 4 >= content.size()) return false;
                        std::string hex = content.substr(pos + 1, 4);
                        int cp = (int)strtol(hex.c_str(), nullptr, 16);
                        key += sam3_codepoint_to_utf8(cp);
                        pos += 4;
                        break;
                    }
                    default:
                        key += content[pos];
                        break;
                }
            } else {
                key += content[pos];
            }
            pos++;
        }
        if (pos >= content.size()) return false;
        pos++;  // skip closing '"'

        // Skip to ':'
        while (pos < content.size() && content[pos] != ':') pos++;
        if (pos >= content.size()) return false;
        pos++;

        // Skip whitespace
        while (pos < content.size() &&
               (content[pos] == ' ' || content[pos] == '\n' ||
                content[pos] == '\r' || content[pos] == '\t'))
            pos++;

        // Read integer
        bool negative = false;
        if (pos < content.size() && content[pos] == '-') {
            negative = true;
            pos++;
        }
        int64_t val = 0;
        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
            val = val * 10 + (content[pos] - '0');
            pos++;
        }
        if (negative) val = -val;

        encoder[key] = (int)val;
    }

    return !encoder.empty();
}

// ── Load merges.txt ──────────────────────────────────────────────────────────

static bool sam3_load_merges(const std::string& path,
                             std::vector<std::pair<std::string, std::string>>& merges,
                             std::unordered_map<std::string, int>& merge_ranks) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    // Skip header line (#version: ...)
    if (!std::getline(f, line)) return false;
    if (!line.empty() && line[0] != '#') {
        // No header — this line IS a merge
        size_t sp = line.find(' ');
        if (sp != std::string::npos) {
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            merge_ranks[sam3_merge_key(a, b)] = (int)merges.size();
            merges.push_back({std::move(a), std::move(b)});
        }
    }

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        merge_ranks[sam3_merge_key(a, b)] = (int)merges.size();
        merges.push_back({std::move(a), std::move(b)});
    }

    return !merges.empty();
}

// ── sam3_load_bpe_vocab ──────────────────────────────────────────────────────

static bool sam3_load_bpe_vocab(sam3_bpe_tokenizer& tok, const std::string& dir) {
    std::string sep(1, '/');
    std::string vocab_path = dir + sep + "vocab.json";
    std::string merges_path = dir + sep + "merges.txt";

    // Load vocabulary
    if (!sam3_parse_vocab_json(vocab_path, tok.encoder)) {
        fprintf(stderr, "%s: failed to load vocab from '%s'\n", __func__, vocab_path.c_str());
        return false;
    }

    // Build decoder (reverse map)
    tok.decoder.clear();
    for (const auto& kv : tok.encoder) {
        tok.decoder[kv.second] = kv.first;
    }

    // Load merges
    if (!sam3_load_merges(merges_path, tok.merges, tok.merge_ranks)) {
        fprintf(stderr, "%s: failed to load merges from '%s'\n", __func__, merges_path.c_str());
        return false;
    }

    // Init byte encoder
    sam3_init_byte_encoder(tok.byte_encoder);

    // Set special tokens
    tok.sot_token = 49406;
    tok.eot_token = 49407;

    fprintf(stderr, "%s: loaded %zu vocab entries, %zu merges\n",
            __func__, tok.encoder.size(), tok.merges.size());
    return true;
}

// ── BPE encode a single word ─────────────────────────────────────────────────

// Split a UTF-8 string into individual unicode characters.
static std::vector<std::string> sam3_utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = sam3_utf8_len((uint8_t)s[i]);
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// Apply BPE merges to a byte-encoded word string.
// Returns space-separated BPE tokens (e.g. "he llo</w>").
static std::string sam3_bpe_encode(sam3_bpe_tokenizer& tok, const std::string& token) {
    auto cit = tok.cache.find(token);
    if (cit != tok.cache.end()) return cit->second;

    // Split into unicode chars, append </w> to last
    std::vector<std::string> word = sam3_utf8_chars(token);
    if (word.empty()) return "";
    word.back() += "</w>";

    if (word.size() == 1) {
        tok.cache[token] = word[0];
        return word[0];
    }

    while (true) {
        // Find pair with lowest merge rank
        int best_rank = INT_MAX;
        std::string best_first, best_second;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = tok.merge_ranks.find(sam3_merge_key(word[i], word[i + 1]));
            if (it != tok.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_first = word[i];
                best_second = word[i + 1];
            }
        }

        if (best_rank == INT_MAX) break;

        // Merge all occurrences of this pair
        std::string merged = best_first + best_second;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() &&
                word[i] == best_first && word[i + 1] == best_second) {
                new_word.push_back(merged);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    // Join with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += ' ';
        result += word[i];
    }
    tok.cache[token] = result;
    return result;
}

// ── Pre-tokenizer (CLIP regex approximation) ─────────────────────────────────

// Splits text into word tokens following the CLIP pattern:
//   <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d
//   | [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+
static std::vector<std::string> sam3_pretokenize(const std::string& text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        uint8_t c = (uint8_t)text[i];

        // Skip whitespace
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            i++;
            continue;
        }

        // Special tokens
        if (i + 15 <= n && text.compare(i, 15, "<|startoftext|>") == 0) {
            tokens.push_back("<|startoftext|>");
            i += 15;
            continue;
        }
        if (i + 13 <= n && text.compare(i, 13, "<|endoftext|>") == 0) {
            tokens.push_back("<|endoftext|>");
            i += 13;
            continue;
        }

        // Contractions (must check before letters since ' isn't a letter)
        if (c == '\'') {
            if (i + 2 <= n) {
                char c2 = text[i + 1];
                if (c2 == 's' || c2 == 't' || c2 == 'm' || c2 == 'd') {
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                    continue;
                }
            }
            if (i + 3 <= n) {
                std::string c3 = text.substr(i + 1, 2);
                if (c3 == "re" || c3 == "ve" || c3 == "ll") {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
            // Fall through — not a contraction
        }

        // Letter sequence
        if (sam3_is_letter(text, i)) {
            size_t start = i;
            while (i < n && sam3_is_letter(text, i)) {
                i += sam3_utf8_len((uint8_t)text[i]);
            }
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Single digit
        if (c >= '0' && c <= '9') {
            tokens.push_back(text.substr(i, 1));
            i++;
            continue;
        }

        // Non-space, non-letter, non-digit sequence
        {
            size_t start = i;
            while (i < n) {
                uint8_t ch = (uint8_t)text[i];
                if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') break;
                if (sam3_is_letter(text, i)) break;
                if (ch >= '0' && ch <= '9') break;
                i++;
            }
            if (i > start) tokens.push_back(text.substr(start, i - start));
        }
    }

    return tokens;
}

// ── sam3_tokenize — full pipeline ────────────────────────────────────────────

// Tokenize text into a fixed-length token ID vector [ctx_len].
// Format: [SOT, bpe_tokens..., EOT, 0, 0, ..., 0]
static std::vector<int32_t> sam3_tokenize(sam3_bpe_tokenizer& tok,
                                          const std::string& text,
                                          int ctx_len) {
    // 1. Lowercase
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }

    // 2. Collapse whitespace, trim
    std::string clean;
    clean.reserve(lower.size());
    bool last_ws = true;
    for (char c : lower) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!last_ws) {
                clean += ' ';
                last_ws = true;
            }
        } else {
            clean += c;
            last_ws = false;
        }
    }
    if (!clean.empty() && clean.back() == ' ') clean.pop_back();

    // 3. Pre-tokenize into word tokens
    auto words = sam3_pretokenize(clean);

    // 4. BPE encode each word
    std::vector<int32_t> ids;
    ids.push_back(tok.sot_token);

    for (const auto& word : words) {
        // Byte-encode: convert each UTF-8 byte through byte_encoder
        std::string encoded;
        for (uint8_t b : word) {
            auto it = tok.byte_encoder.find(b);
            if (it != tok.byte_encoder.end()) {
                encoded += it->second;
            }
        }

        // BPE
        std::string bpe_result = sam3_bpe_encode(tok, encoded);

        // Split on spaces → look up each token
        size_t start = 0;
        while (start < bpe_result.size()) {
            size_t end = bpe_result.find(' ', start);
            if (end == std::string::npos) end = bpe_result.size();
            std::string bpe_tok = bpe_result.substr(start, end - start);

            auto eit = tok.encoder.find(bpe_tok);
            if (eit != tok.encoder.end()) {
                ids.push_back(eit->second);
            }
            // Unknown tokens are silently dropped (matches CLIP behavior
            // where all byte sequences are in the vocab)

            start = end + 1;
        }
    }

    ids.push_back(tok.eot_token);

    // 5. Truncate (keep SOT at front, force EOT at end)
    if ((int)ids.size() > ctx_len) {
        ids.resize(ctx_len);
        ids.back() = tok.eot_token;
    }

    // 6. Pad with 0 to ctx_len
    ids.resize(ctx_len, 0);

    return ids;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model loading — internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

static bool sam3_load_hparams(std::ifstream& fin, sam3_hparams& hp) {
    auto rd = [&](int32_t& v) { fin.read(reinterpret_cast<char*>(&v), 4); };
    rd(hp.img_size);
    rd(hp.patch_size);
    rd(hp.vit_embed_dim);
    rd(hp.vit_depth);
    rd(hp.vit_num_heads);
    int32_t mlp_ratio_x1000;
    rd(mlp_ratio_x1000);
    hp.vit_mlp_dim = static_cast<int32_t>(hp.vit_embed_dim * (mlp_ratio_x1000 / 1000.0f));
    rd(hp.vit_window_size);
    rd(hp.n_global_attn);
    for (int i = 0; i < hp.n_global_attn && i < 4; ++i) {
        rd(hp.global_attn_idx[i]);
    }
    rd(hp.text_width);
    rd(hp.text_heads);
    rd(hp.text_layers);
    rd(hp.text_ctx_len);
    rd(hp.text_vocab_size);
    rd(hp.text_out_dim);
    rd(hp.neck_dim);
    rd(hp.fenc_layers);
    rd(hp.fenc_heads);
    rd(hp.fenc_ffn_dim);
    rd(hp.ddec_layers);
    rd(hp.ddec_heads);
    rd(hp.ddec_ffn_dim);
    rd(hp.ddec_num_queries);
    rd(hp.geom_layers);
    rd(hp.n_presence_tokens);
    rd(hp.n_geom_queries);
    rd(hp.sam_embed_dim);
    rd(hp.sam_dec_depth);
    rd(hp.sam_n_multimask);
    rd(hp.sam_iou_head_depth);
    rd(hp.mem_out_dim);
    rd(hp.mem_attn_layers);
    rd(hp.num_maskmem);
    rd(hp.max_obj_ptrs);
    rd(hp.n_amb_experts);
    return !fin.fail();
}

static void sam3_print_hparams(const sam3_hparams& hp) {
    fprintf(stderr, "  img_size       = %d\n", hp.img_size);
    fprintf(stderr, "  patch_size     = %d\n", hp.patch_size);
    fprintf(stderr, "  vit_embed_dim  = %d\n", hp.vit_embed_dim);
    fprintf(stderr, "  vit_depth      = %d\n", hp.vit_depth);
    fprintf(stderr, "  vit_num_heads  = %d\n", hp.vit_num_heads);
    fprintf(stderr, "  vit_mlp_dim    = %d\n", hp.vit_mlp_dim);
    fprintf(stderr, "  vit_window     = %d\n", hp.vit_window_size);
    fprintf(stderr, "  text_width     = %d\n", hp.text_width);
    fprintf(stderr, "  text_layers    = %d\n", hp.text_layers);
    fprintf(stderr, "  neck_dim       = %d\n", hp.neck_dim);
    fprintf(stderr, "  fenc_layers    = %d\n", hp.fenc_layers);
    fprintf(stderr, "  ddec_layers    = %d\n", hp.ddec_layers);
    fprintf(stderr, "  ddec_queries   = %d\n", hp.ddec_num_queries);
    fprintf(stderr, "  sam_embed_dim  = %d\n", hp.sam_embed_dim);
    fprintf(stderr, "  mem_attn_lyrs  = %d\n", hp.mem_attn_layers);
    fprintf(stderr, "  num_maskmem    = %d\n", hp.num_maskmem);
}

// Register all tensor names in the model struct so we can look them up by name
// when loading from the binary file. This creates ggml tensors with no_alloc
// (metadata only) and populates model.tensors.
static void sam3_register_tensors(sam3_model& model) {
    const auto& hp = model.hparams;
    auto& tensors = model.tensors;
    auto ctx = model.ctx;

    auto T1 = [&](const std::string& name, int64_t d0) -> ggml_tensor* {
        auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    const ggml_type WTYPE = model.weight_type;

    auto T2 = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        auto* t = ggml_new_tensor_2d(ctx, WTYPE, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3 = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor* {
        auto* t = ggml_new_tensor_3d(ctx, WTYPE, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4 = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        auto* t = ggml_new_tensor_4d(ctx, WTYPE, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    // Always f32 (for embeddings, biases, norms)
    auto T1f = T1;
    auto T2f = [&](const std::string& name, int64_t d0, int64_t d1) -> ggml_tensor* {
        auto* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor* {
        auto* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4f = [&](const std::string& name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor* {
        auto* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };

    const int E = hp.vit_embed_dim;      // 1024
    const int D = hp.neck_dim;           // 256
    const int TW = hp.text_width;        // 1024
    const int MLP = hp.vit_mlp_dim;      // 4736
    const int FFN = hp.fenc_ffn_dim;     // 2048
    const int NQ = hp.ddec_num_queries;  // 200
    const int MD = hp.mem_out_dim;       // 64
    const int H = hp.n_img_embd();       // 72

    // ── ViT backbone ─────────────────────────────────────────────────────
    model.vit.blocks.resize(hp.vit_depth);

    model.vit.patch_embed_w = T4("vit.patch_embed.proj.weight", hp.patch_size, hp.patch_size, 3, E);
    // pos_embed: Hiera stores [1, 24, 24, 1024] at pretrained resolution (no cls token).
    // Conversion script writes reversed dims → ggml [E, 24, 24, 1].
    // Tiled 3x at runtime to [E, 72, 72, 1].
    {
        const int pretrained_grid = hp.img_size / hp.patch_size / 3;  // 1008/14/3 = 24
        model.vit.pos_embed = T4f("vit.pos_embed", E, pretrained_grid, pretrained_grid, 1);
    }
    model.vit.ln_pre_w = T1f("vit.ln_pre.weight", E);
    model.vit.ln_pre_b = T1f("vit.ln_pre.bias", E);

    for (int i = 0; i < hp.vit_depth; ++i) {
        auto& blk = model.vit.blocks[i];
        auto p = "vit.blocks." + std::to_string(i);
        blk.norm1_w = T1f(p + ".norm1.weight", E);
        blk.norm1_b = T1f(p + ".norm1.bias", E);
        blk.qkv_w = T2(p + ".attn.qkv.weight", E, 3 * E);
        blk.qkv_b = T1f(p + ".attn.qkv.bias", 3 * E);
        blk.proj_w = T2(p + ".attn.proj.weight", E, E);
        blk.proj_b = T1f(p + ".attn.proj.bias", E);
        blk.norm2_w = T1f(p + ".norm2.weight", E);
        blk.norm2_b = T1f(p + ".norm2.bias", E);
        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", E, MLP);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", MLP);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", MLP, E);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", E);

        // RoPE freqs_cis: [N, 32, 2] where N=5184 for global, 576 for window
        int64_t rope_n = hp.is_global_attn(i) ? hp.n_img_tokens() : (hp.vit_window_size * hp.vit_window_size);
        blk.freqs_cis = T3f(p + ".attn.freqs_cis", 2, 32, rope_n);
    }

    // ── Neck (detector + tracker) ────────────────────────────────────────
    // ggml conv2d kernel: [kW, kH, Cin, Cout]
    // ggml conv_transpose kernel: [kW, kH, Cout, Cin]
    // PyTorch Conv2d(Cin, Cout, k) weight: [Cout, Cin, kH, kW] → ggml [kW, kH, Cin, Cout]
    // PyTorch ConvTranspose2d(Cin, Cout, k) weight: [Cin, Cout, kH, kW] → ggml [kW, kH, Cout, Cin]
    auto register_neck = [&](sam3_neck& neck, const std::string& prefix) {
        // scale 0 (4x): ConvTranspose(E→512, k=2, s=2), GELU, ConvTranspose(512→D, k=2, s=2), Conv1x1(D→D), Conv3x3(D→D)
        neck.scales[0].deconv1_w = T4(prefix + "0.dconv_2x2_0.weight", 2, 2, 512, E);  // [kW, kH, Cout=512, Cin=E]
        neck.scales[0].deconv1_b = T1f(prefix + "0.dconv_2x2_0.bias", 512);
        neck.scales[0].deconv2_w = T4(prefix + "0.dconv_2x2_1.weight", 2, 2, D, 512);  // [kW, kH, Cout=D, Cin=512]
        neck.scales[0].deconv2_b = T1f(prefix + "0.dconv_2x2_1.bias", D);
        neck.scales[0].conv1x1_w = T4(prefix + "0.conv_1x1.weight", 1, 1, D, D);  // Conv2d(D→D)
        neck.scales[0].conv1x1_b = T1f(prefix + "0.conv_1x1.bias", D);
        neck.scales[0].conv3x3_w = T4(prefix + "0.conv_3x3.weight", 3, 3, D, D);  // Conv2d(D→D)
        neck.scales[0].conv3x3_b = T1f(prefix + "0.conv_3x3.bias", D);

        // scale 1 (2x): ConvTranspose(E→512, k=2, s=2), Conv1x1(512→D), Conv3x3(D→D)
        neck.scales[1].deconv1_w = T4(prefix + "1.dconv_2x2.weight", 2, 2, 512, E);  // ConvTranspose
        neck.scales[1].deconv1_b = T1f(prefix + "1.dconv_2x2.bias", 512);
        neck.scales[1].conv1x1_w = T4(prefix + "1.conv_1x1.weight", 1, 1, 512, D);  // Conv2d(512→D): Cin=512, Cout=D
        neck.scales[1].conv1x1_b = T1f(prefix + "1.conv_1x1.bias", D);
        neck.scales[1].conv3x3_w = T4(prefix + "1.conv_3x3.weight", 3, 3, D, D);
        neck.scales[1].conv3x3_b = T1f(prefix + "1.conv_3x3.bias", D);

        // scale 2 (1x): Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[2].conv1x1_w = T4(prefix + "2.conv_1x1.weight", 1, 1, E, D);  // Conv2d(E→D): Cin=E, Cout=D
        neck.scales[2].conv1x1_b = T1f(prefix + "2.conv_1x1.bias", D);
        neck.scales[2].conv3x3_w = T4(prefix + "2.conv_3x3.weight", 3, 3, D, D);
        neck.scales[2].conv3x3_b = T1f(prefix + "2.conv_3x3.bias", D);

        // scale 3 (0.5x): MaxPool(k=2, s=2), Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[3].conv1x1_w = T4(prefix + "3.conv_1x1.weight", 1, 1, E, D);
        neck.scales[3].conv1x1_b = T1f(prefix + "3.conv_1x1.bias", D);
        neck.scales[3].conv3x3_w = T4(prefix + "3.conv_3x3.weight", 3, 3, D, D);
        neck.scales[3].conv3x3_b = T1f(prefix + "3.conv_3x3.bias", D);
    };
    register_neck(model.neck_det, "neck.det.");
    register_neck(model.neck_trk, "neck.trk.");

    // ── Text encoder ─────────────────────────────────────────────────────
    model.text_enc.blocks.resize(hp.text_layers);
    model.text_enc.token_embed_w = T2f("text.token_embed.weight", TW, hp.text_vocab_size);
    model.text_enc.pos_embed = T2f("text.pos_embed", TW, hp.text_ctx_len);
    model.text_enc.ln_final_w = T1f("text.ln_final.weight", TW);
    model.text_enc.ln_final_b = T1f("text.ln_final.bias", TW);
    model.text_enc.resizer_w = T2("text.resizer.weight", TW, hp.text_out_dim);
    model.text_enc.resizer_b = T1f("text.resizer.bias", hp.text_out_dim);
    // text.text_projection is intentionally not registered — the conversion
    // script skips it and the loader rejects unknown tensors. See struct comment.

    for (int i = 0; i < hp.text_layers; ++i) {
        auto& blk = model.text_enc.blocks[i];
        auto p = "text.blocks." + std::to_string(i);
        blk.attn_in_proj_w = T2(p + ".attn.in_proj.weight", TW, 3 * TW);
        blk.attn_in_proj_b = T1f(p + ".attn.in_proj.bias", 3 * TW);
        blk.attn_out_proj_w = T2(p + ".attn.out_proj.weight", TW, TW);
        blk.attn_out_proj_b = T1f(p + ".attn.out_proj.bias", TW);
        blk.ln1_w = T1f(p + ".ln_1.weight", TW);
        blk.ln1_b = T1f(p + ".ln_1.bias", TW);
        blk.ln2_w = T1f(p + ".ln_2.weight", TW);
        blk.ln2_b = T1f(p + ".ln_2.bias", TW);
        blk.mlp_fc1_w = T2(p + ".mlp.fc1.weight", TW, TW * 4);
        blk.mlp_fc1_b = T1f(p + ".mlp.fc1.bias", TW * 4);
        blk.mlp_fc2_w = T2(p + ".mlp.fc2.weight", TW * 4, TW);
        blk.mlp_fc2_b = T1f(p + ".mlp.fc2.bias", TW);
    }

    // ── Fusion encoder ───────────────────────────────────────────────────
    model.fenc.layers.resize(hp.fenc_layers);
    for (int i = 0; i < hp.fenc_layers; ++i) {
        auto& ly = model.fenc.layers[i];
        auto p = "fenc.layers." + std::to_string(i);
        // self-attention
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_kv_w = nullptr;  // fused in_proj for MHA
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── DETR decoder ─────────────────────────────────────────────────────
    model.ddec.layers.resize(hp.ddec_layers);
    model.ddec.query_embed = T2f("ddec.query_embed.weight", D, NQ);
    model.ddec.presence_token = T2f("ddec.presence_token.weight", D, 1);

    // Reference points, norms, bbox embed, ref_point_head, boxRPB, presence_head
    // These use the exact checkpoint names after renaming
    tensors["ddec.reference_points.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, NQ);
    ggml_set_name(tensors["ddec.reference_points.weight"], "ddec.reference_points.weight");
    tensors["ddec.norm.weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_set_name(tensors["ddec.norm.weight"], "ddec.norm.weight");
    tensors["ddec.norm.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_set_name(tensors["ddec.norm.bias"], "ddec.norm.bias");

    // bbox_embed MLP (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 4 : D;
        auto bp = "ddec.bbox_embed.layers." + std::to_string(j);
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, WTYPE, D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }

    // ref_point_head MLP (2 layers: 512→256→256)
    tensors["ddec.ref_point_head.layers.0.weight"] = ggml_new_tensor_2d(ctx, WTYPE, 512, D);
    tensors["ddec.ref_point_head.layers.0.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    tensors["ddec.ref_point_head.layers.1.weight"] = ggml_new_tensor_2d(ctx, WTYPE, D, D);
    tensors["ddec.ref_point_head.layers.1.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    for (auto& kv : std::vector<std::string>{
             "ddec.ref_point_head.layers.0.weight", "ddec.ref_point_head.layers.0.bias",
             "ddec.ref_point_head.layers.1.weight", "ddec.ref_point_head.layers.1.bias"})
        ggml_set_name(tensors[kv], kv.c_str());

    // boxRPB MLPs (x and y, each 2 layers)
    for (const auto& axis : {"x", "y"}) {
        auto bp = std::string("ddec.boxRPB_embed_") + axis;
        tensors[bp + ".layers.0.weight"] = ggml_new_tensor_2d(ctx, WTYPE, 2, D);
        tensors[bp + ".layers.0.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        tensors[bp + ".layers.1.weight"] = ggml_new_tensor_2d(ctx, WTYPE, D, hp.ddec_heads);
        tensors[bp + ".layers.1.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.ddec_heads);
        for (int j = 0; j < 2; ++j) {
            auto l = bp + ".layers." + std::to_string(j);
            ggml_set_name(tensors[l + ".weight"], (l + ".weight").c_str());
            ggml_set_name(tensors[l + ".bias"], (l + ".bias").c_str());
        }
    }

    // presence_token_head MLP (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out = (j == 2) ? 1 : D;
        auto bp = "ddec.presence_token_head.layers." + std::to_string(j);
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, WTYPE, D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }
    tensors["ddec.presence_token_out_norm.weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    tensors["ddec.presence_token_out_norm.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_set_name(tensors["ddec.presence_token_out_norm.weight"], "ddec.presence_token_out_norm.weight");
    ggml_set_name(tensors["ddec.presence_token_out_norm.bias"], "ddec.presence_token_out_norm.bias");

    // DETR decoder layers
    for (int i = 0; i < hp.ddec_layers; ++i) {
        auto& ly = model.ddec.layers[i];
        auto p = "ddec.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);

        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);

        ly.ca_text_q_w = T2(p + ".ca_text.in_proj_weight", D, 3 * D);
        ly.ca_text_q_b = T1f(p + ".ca_text.in_proj_bias", 3 * D);
        ly.ca_text_out_w = T2(p + ".ca_text.out_proj.weight", D, D);
        ly.ca_text_out_b = T1f(p + ".ca_text.out_proj.bias", D);
        ly.norm3_w = T1f(p + ".norm_ca_text.weight", D);
        ly.norm3_b = T1f(p + ".norm_ca_text.bias", D);

        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm4_w = T1f(p + ".norm3.weight", D);
        ly.norm4_b = T1f(p + ".norm3.bias", D);
    }

    // ── DotProductScoring ────────────────────────────────────────────────
    auto reg = [&](const std::string& n, int64_t d0, int64_t d1, bool is_f32 = false) {
        auto* t = ggml_new_tensor_2d(ctx, is_f32 ? GGML_TYPE_F32 : WTYPE, d0, d1);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg1 = [&](const std::string& n, int64_t d0) {
        auto* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg4 = [&](const std::string& n, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
        auto* t = ggml_new_tensor_4d(ctx, WTYPE, d0, d1, d2, d3);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };

    reg("scoring.prompt_proj.weight", D, D);
    reg1("scoring.prompt_proj.bias", D);
    reg("scoring.hs_proj.weight", D, D);
    reg1("scoring.hs_proj.bias", D);
    reg("scoring.prompt_mlp.layers.0.weight", D, FFN);
    reg1("scoring.prompt_mlp.layers.0.bias", FFN);
    reg("scoring.prompt_mlp.layers.1.weight", FFN, D);
    reg1("scoring.prompt_mlp.layers.1.bias", D);
    reg1("scoring.prompt_mlp.out_norm.weight", D);
    reg1("scoring.prompt_mlp.out_norm.bias", D);

    // ── Geometry encoder ───────────────────────────────────────────────────
    model.geom_enc.layers.resize(hp.geom_layers);

    model.geom_enc.point_proj_w = T2("geom.points_direct_project.weight", 2, D);
    model.geom_enc.point_proj_b = T1f("geom.points_direct_project.bias", D);
    model.geom_enc.box_proj_w = T2("geom.boxes_direct_project.weight", 4, D);
    model.geom_enc.box_proj_b = T1f("geom.boxes_direct_project.bias", D);
    model.geom_enc.type_embed = T2f("geom.label_embed.weight", D, 2);
    model.geom_enc.cls_token = T2f("geom.cls_embed.weight", D, 1);
    model.geom_enc.post_proj_w = T2("geom.final_proj.weight", D, D);
    model.geom_enc.post_proj_b = T1f("geom.final_proj.bias", D);

    // Points and boxes pool/pos projections
    reg("geom.points_pool_project.weight", D, D);
    reg1("geom.points_pool_project.bias", D);
    reg("geom.points_pos_enc_project.weight", D, D);
    reg1("geom.points_pos_enc_project.bias", D);
    reg4("geom.boxes_pool_project.weight", 7, 7, D, D);
    reg1("geom.boxes_pool_project.bias", D);
    reg("geom.boxes_pos_enc_project.weight", 258, D);
    reg1("geom.boxes_pos_enc_project.bias", D);

    // Norms
    reg1("geom.norm.weight", D);
    reg1("geom.norm.bias", D);
    reg1("geom.encode_norm.weight", D);
    reg1("geom.encode_norm.bias", D);
    reg1("geom.img_pre_norm.weight", D);
    reg1("geom.img_pre_norm.bias", D);

    for (int i = 0; i < hp.geom_layers; ++i) {
        auto& ly = model.geom_enc.layers[i];
        auto p = "geom.layers." + std::to_string(i);
        ly.sa_in_proj_w = T2(p + ".sa.in_proj_weight", D, 3 * D);
        ly.sa_in_proj_b = T1f(p + ".sa.in_proj_bias", 3 * D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        ly.ca_q_w = T2(p + ".ca.in_proj_weight", D, 3 * D);
        ly.ca_q_b = T1f(p + ".ca.in_proj_bias", 3 * D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── Segmentation head ────────────────────────────────────────────────
    // Pixel decoder (3 conv layers + norms)
    for (int i = 0; i < 3; ++i) {
        auto si = std::to_string(i);
        model.seg_head.up_conv_w[i] = T4("seg.pixel_decoder.conv_layers." + si + ".weight", 3, 3, D, D);
        model.seg_head.up_conv_b[i] = T1f("seg.pixel_decoder.conv_layers." + si + ".bias", D);
        model.seg_head.up_norm_w[i] = T1f("seg.pixel_decoder.norms." + si + ".weight", D);
        model.seg_head.up_norm_b[i] = T1f("seg.pixel_decoder.norms." + si + ".bias", D);
    }

    // Mask predictor (3-layer MLP: 256→256→256→256)
    for (int j = 0; j < 3; ++j) {
        auto bp = "seg.mask_predictor.mask_embed.layers." + std::to_string(j);
        model.seg_head.mask_embed_w = T2(bp + ".weight", D, D);  // overwritten but last one
        model.seg_head.mask_embed_b = T1f(bp + ".bias", D);
    }
    // Re-register properly: all 3 layers with unique names are already in tensors map
    // The struct only has one pointer — use the tensors map at runtime
    // For now, just ensure all 6 tensors are registered (they are via the loop above —
    // each T2/T1f call registers under unique names)

    // Cross-attention to prompt
    model.seg_head.ca_prompt_q_w = T2("seg.cross_attend_prompt.in_proj_weight", D, 3 * D);
    model.seg_head.ca_prompt_q_b = T1f("seg.cross_attend_prompt.in_proj_bias", 3 * D);
    model.seg_head.ca_prompt_out_w = T2("seg.cross_attend_prompt.out_proj.weight", D, D);
    model.seg_head.ca_prompt_out_b = T1f("seg.cross_attend_prompt.out_proj.bias", D);

    // Cross-attn norm
    reg1("seg.cross_attn_norm.weight", D);
    reg1("seg.cross_attn_norm.bias", D);

    // Instance and semantic seg heads (Conv 1x1)
    reg4("seg.instance_seg_head.weight", 1, 1, D, D);
    reg1("seg.instance_seg_head.bias", D);
    reg4("seg.semantic_seg_head.weight", 1, 1, D, 1);
    reg1("seg.semantic_seg_head.bias", 1);

    // ── SAM prompt encoder ───────────────────────────────────────────────
    model.sam_pe.pe_gaussian = T2f("sam_pe.pe_gaussian", 2, 128);
    for (int i = 0; i < 4; ++i)
        model.sam_pe.point_embed[i] = T2f("sam_pe.point_embeddings." + std::to_string(i) + ".weight", D, 1);
    model.sam_pe.not_a_point_embed = T2f("sam_pe.not_a_point_embed.weight", D, 1);
    model.sam_pe.no_mask_embed = T2f("sam_pe.no_mask_embed.weight", D, 1);

    // mask_downscaling: sequential with numeric indices
    model.sam_pe.mask_ds_conv_w[0] = T4("sam_pe.mask_ds.0.weight", 2, 2, 1, 4);
    model.sam_pe.mask_ds_conv_b[0] = T1f("sam_pe.mask_ds.0.bias", 4);
    model.sam_pe.mask_ds_norm_w[0] = T1f("sam_pe.mask_ds.1.weight", 4);
    model.sam_pe.mask_ds_norm_b[0] = T1f("sam_pe.mask_ds.1.bias", 4);
    model.sam_pe.mask_ds_conv_w[1] = T4("sam_pe.mask_ds.3.weight", 2, 2, 4, 16);
    model.sam_pe.mask_ds_conv_b[1] = T1f("sam_pe.mask_ds.3.bias", 16);
    model.sam_pe.mask_ds_norm_w[1] = T1f("sam_pe.mask_ds.4.weight", 16);
    model.sam_pe.mask_ds_norm_b[1] = T1f("sam_pe.mask_ds.4.bias", 16);
    model.sam_pe.mask_ds_conv_w[2] = T4("sam_pe.mask_ds.6.weight", 1, 1, 16, D);
    model.sam_pe.mask_ds_conv_b[2] = T1f("sam_pe.mask_ds.6.bias", D);

    // ── SAM mask decoder ─────────────────────────────────────────────────
    model.sam_dec.iou_token = T2f("sam_dec.iou_token.weight", D, 1);
    model.sam_dec.mask_tokens = T2f("sam_dec.mask_tokens.weight", D, 4);
    model.sam_dec.obj_score_token = T2f("sam_dec.obj_score_token.weight", D, 1);

    model.sam_dec.twoway_blocks.resize(hp.sam_dec_depth);
    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        auto& blk = model.sam_dec.twoway_blocks[i];
        auto p = "sam_dec.twoway." + std::to_string(i);

        auto reg_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
            a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
            a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
            a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
            a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
            a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
            a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
            a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
            a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
        };

        reg_attn(blk.self_attn, p + ".sa", D, D);
        reg_attn(blk.ca_tok2img, p + ".cross_attn_token_to_image", D, 128);
        reg_attn(blk.ca_img2tok, p + ".cross_attn_image_to_token", D, 128);

        blk.norm1_w = T1f(p + ".norm1.weight", D);
        blk.norm1_b = T1f(p + ".norm1.bias", D);
        blk.norm2_w = T1f(p + ".norm2.weight", D);
        blk.norm2_b = T1f(p + ".norm2.bias", D);
        blk.norm3_w = T1f(p + ".norm3.weight", D);
        blk.norm3_b = T1f(p + ".norm3.bias", D);
        blk.norm4_w = T1f(p + ".norm4.weight", D);
        blk.norm4_b = T1f(p + ".norm4.bias", D);

        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", D, FFN);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", FFN);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", FFN, D);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", D);
    }

    // final attention
    auto reg_sam_attn = [&](sam3_sam_attn& a, const std::string& pfx, int in_dim, int out_dim) {
        a.q_w = T2(pfx + ".q_proj.weight", in_dim, out_dim);
        a.q_b = T1f(pfx + ".q_proj.bias", out_dim);
        a.k_w = T2(pfx + ".k_proj.weight", in_dim, out_dim);
        a.k_b = T1f(pfx + ".k_proj.bias", out_dim);
        a.v_w = T2(pfx + ".v_proj.weight", in_dim, out_dim);
        a.v_b = T1f(pfx + ".v_proj.bias", out_dim);
        a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
        a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
    };
    reg_sam_attn(model.sam_dec.final_attn, "sam_dec.final_attn", D, 128);
    model.sam_dec.final_norm_w = T1f("sam_dec.final_norm.weight", D);
    model.sam_dec.final_norm_b = T1f("sam_dec.final_norm.bias", D);

    // upscaling
    model.sam_dec.up1_w = T4("sam_dec.upscale.0.weight", 2, 2, 64, D);
    model.sam_dec.up1_b = T1f("sam_dec.upscale.0.bias", 64);
    model.sam_dec.up1_norm_w = T1f("sam_dec.upscale.1.weight", 64);
    model.sam_dec.up1_norm_b = T1f("sam_dec.upscale.1.bias", 64);
    model.sam_dec.up2_w = T4("sam_dec.upscale.3.weight", 2, 2, 32, 64);
    model.sam_dec.up2_b = T1f("sam_dec.upscale.3.bias", 32);

    // high-res feature convolutions
    model.sam_dec.conv_s0_w = T4("sam_dec.conv_s0.weight", 1, 1, D, 32);
    model.sam_dec.conv_s0_b = T1f("sam_dec.conv_s0.bias", 32);
    model.sam_dec.conv_s1_w = T4("sam_dec.conv_s1.weight", 1, 1, D, 64);
    model.sam_dec.conv_s1_b = T1f("sam_dec.conv_s1.bias", 64);

    // hypernetwork MLPs (4 × 3 layers: 256→256→256→32)
    for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < 3; ++j) {
            int in_d = D, out_d = (j == 2) ? 32 : D;
            auto bp = "sam_dec.hyper." + std::to_string(m) + ".layers." + std::to_string(j);
            model.sam_dec.hyper_w[m][j] = T2(bp + ".weight", in_d, out_d);
            model.sam_dec.hyper_b[m][j] = T1f(bp + ".bias", out_d);
        }
    }

    // IoU prediction head (3 layers: 256→256→256→4)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 4 : D;
        auto bp = "sam_dec.iou_prediction_head.layers." + std::to_string(j);
        model.sam_dec.iou_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.iou_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // object score head (3 layers: 256→256→256→1)
    for (int j = 0; j < 3; ++j) {
        int out_d = (j == 2) ? 1 : D;
        auto bp = "sam_dec.pred_obj_score_head.layers." + std::to_string(j);
        model.sam_dec.obj_head_w[j] = T2(bp + ".weight", D, out_d);
        model.sam_dec.obj_head_b[j] = T1f(bp + ".bias", out_d);
    }

    // ── Memory encoder ───────────────────────────────────────────────────
    // mask_downsampler: sequential encoder.{0,1,3,4,6,7,9,10,12}
    int ds_channels[] = {1, 4, 16, 64, 256};
    int ds_indices[] = {0, 3, 6, 9, 12};
    int norm_indices[] = {1, 4, 7, 10};
    for (int s = 0; s < 4; ++s) {
        auto si = std::to_string(ds_indices[s]);
        model.mem_enc.ds_conv_w[s] = T4("mem_enc.ds." + si + ".weight", 3, 3, ds_channels[s], ds_channels[s + 1]);
        model.mem_enc.ds_conv_b[s] = T1f("mem_enc.ds." + si + ".bias", ds_channels[s + 1]);
        auto ni = std::to_string(norm_indices[s]);
        model.mem_enc.ds_norm_w[s] = T1f("mem_enc.ds." + ni + ".weight", ds_channels[s + 1]);
        model.mem_enc.ds_norm_b[s] = T1f("mem_enc.ds." + ni + ".bias", ds_channels[s + 1]);
    }
    model.mem_enc.ds_conv_w[4] = T4("mem_enc.ds.12.weight", 1, 1, D, D);
    model.mem_enc.ds_conv_b[4] = T1f("mem_enc.ds.12.bias", D);

    model.mem_enc.pix_proj_w = T4("mem_enc.pix_feat_proj.weight", 1, 1, D, D);
    model.mem_enc.pix_proj_b = T1f("mem_enc.pix_feat_proj.bias", D);

    // fuser CXBlocks
    for (int i = 0; i < 2; ++i) {
        auto p = "mem_enc.fuser." + std::to_string(i);
        model.mem_enc.fuser_dw_w[i] = T4(p + ".dwconv.weight", 7, 7, 1, D);  // groups=256
        model.mem_enc.fuser_dw_b[i] = T1f(p + ".dwconv.bias", D);
        model.mem_enc.fuser_norm_w[i] = T1f(p + ".norm.weight", D);
        model.mem_enc.fuser_norm_b[i] = T1f(p + ".norm.bias", D);
        model.mem_enc.fuser_fc1_w[i] = T2(p + ".pwconv1.weight", D, 1024);
        model.mem_enc.fuser_fc1_b[i] = T1f(p + ".pwconv1.bias", 1024);
        model.mem_enc.fuser_fc2_w[i] = T2(p + ".pwconv2.weight", 1024, D);
        model.mem_enc.fuser_fc2_b[i] = T1f(p + ".pwconv2.bias", D);
        model.mem_enc.fuser_gamma[i] = T1f(p + ".gamma", D);
    }

    model.mem_enc.out_proj_w = T4("mem_enc.out_proj.weight", 1, 1, D, MD);
    model.mem_enc.out_proj_b = T1f("mem_enc.out_proj.bias", MD);

    // temporal pos encodings
    model.mem_enc.tpos[0] = T4f("mem_enc.tpos_enc", MD, 1, 1, hp.num_maskmem);

    // ── Memory attention ─────────────────────────────────────────────────
    model.mem_attn.layers.resize(hp.mem_attn_layers);
    reg1("mem_attn.norm.weight", D);
    reg1("mem_attn.norm.bias", D);

    for (int i = 0; i < hp.mem_attn_layers; ++i) {
        auto& ly = model.mem_attn.layers[i];
        auto p = "mem_attn.layers." + std::to_string(i);
        // self-attention (RoPE, 1 head, 256-dim)
        ly.sa_q_w = T2(p + ".sa.q_proj.weight", D, D);
        ly.sa_q_b = T1f(p + ".sa.q_proj.bias", D);
        ly.sa_k_w = T2(p + ".sa.k_proj.weight", D, D);
        ly.sa_k_b = T1f(p + ".sa.k_proj.bias", D);
        ly.sa_v_w = T2(p + ".sa.v_proj.weight", D, D);
        ly.sa_v_b = T1f(p + ".sa.v_proj.bias", D);
        ly.sa_out_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention (kv_in_dim=64) — renamed from cross_attn_image → ca
        ly.ca_q_w = T2(p + ".ca.q_proj.weight", D, D);
        ly.ca_q_b = T1f(p + ".ca.q_proj.bias", D);
        ly.ca_k_w = T2(p + ".ca.k_proj.weight", MD, D);
        ly.ca_k_b = T1f(p + ".ca.k_proj.bias", D);
        ly.ca_v_w = T2(p + ".ca.v_proj.weight", MD, D);
        ly.ca_v_b = T1f(p + ".ca.v_proj.bias", D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w = T1f(p + ".norm3.weight", D);
        ly.norm3_b = T1f(p + ".norm3.bias", D);
    }

    // ── Object pointer projection ────────────────────────────────────────
    for (int j = 0; j < 3; ++j) {
        auto bp = "obj_ptr_proj.layers." + std::to_string(j);
        model.obj_ptr_proj_w[j] = T2(bp + ".weight", D, D);
        model.obj_ptr_proj_b[j] = T1f(bp + ".bias", D);
    }
    model.no_obj_ptr = T2f("no_obj_ptr", D, 1);
    model.obj_ptr_tpos_w = T2("obj_ptr_tpos_proj.weight", D, MD);
    model.obj_ptr_tpos_b = T1f("obj_ptr_tpos_proj.bias", MD);

    // standalone tracker params
    // standalone tracker parameters
    T3f("no_mem_embed", D, 1, 1);           // [1, 1, 256]
    T3f("no_mem_pos_enc", D, 1, 1);         // [1, 1, 256]
    T2f("no_obj_embed_spatial", MD, 1);     // [1, 64]
    T4f("trk_mask_ds.weight", 4, 4, 1, 1);  // nn.Conv2d(1,1,4,4): [1,1,4,4]
    T1f("trk_mask_ds.bias", 1);
}

// Load tensors from the binary file into the already-registered ggml tensors
static bool sam3_load_tensors(std::ifstream& fin, sam3_model& model) {
    int n_loaded = 0;
    while (fin.peek() != EOF) {
        int32_t n_dims, name_len, dtype;
        fin.read(reinterpret_cast<char*>(&n_dims), 4);
        fin.read(reinterpret_cast<char*>(&name_len), 4);
        fin.read(reinterpret_cast<char*>(&dtype), 4);
        if (fin.fail()) break;

        // Read shape (reversed in file)
        std::vector<int64_t> shape(n_dims);
        for (int i = 0; i < n_dims; ++i) {
            int32_t d;
            fin.read(reinterpret_cast<char*>(&d), 4);
            shape[i] = d;
        }

        // Read name
        std::string name(name_len, '\0');
        fin.read(&name[0], name_len);

        // Skip to 32-byte alignment
        size_t pos = fin.tellg();
        size_t pad = (32 - pos % 32) % 32;
        if (pad > 0) fin.seekg(pad, std::ios::cur);

        // Look up tensor — every tensor in the file must be registered
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in file (not registered by model)\n",
                    __func__, name.c_str());
            return false;
        }

        auto* tensor = it->second;

        // Compute data size from file dtype
        int64_t n_el = 1;
        for (auto d : shape) n_el *= d;
        size_t file_elem_size = (dtype == 1 /*f16*/) ? 2 : 4;
        size_t bytes = n_el * file_elem_size;

        // Read into a temporary CPU buffer, then copy to backend
        std::vector<char> buf(bytes);
        fin.read(buf.data(), bytes);
        if (fin.fail()) {
            fprintf(stderr, "%s: failed to read tensor '%s'\n", __func__, name.c_str());
            return false;
        }

        // If the file dtype matches the registered tensor type, copy directly.
        // Otherwise, convert f32 ↔ f16 as needed.
        ggml_type file_type = (dtype == 1 /*f16*/) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (file_type == tensor->type) {
            ggml_backend_tensor_set(tensor, buf.data(), 0, bytes);
        } else if (file_type == GGML_TYPE_F16 && tensor->type == GGML_TYPE_F32) {
            // Convert f16 → f32
            std::vector<float> f32_buf(n_el);
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(buf.data()),
                                  f32_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f32_buf.data(), 0, n_el * sizeof(float));
        } else if (file_type == GGML_TYPE_F32 && tensor->type == GGML_TYPE_F16) {
            // Convert f32 → f16
            std::vector<ggml_fp16_t> f16_buf(n_el);
            ggml_fp32_to_fp16_row(reinterpret_cast<const float*>(buf.data()),
                                  f16_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f16_buf.data(), 0, n_el * sizeof(ggml_fp16_t));
        } else {
            fprintf(stderr, "%s: unsupported type conversion for '%s'\n", __func__, name.c_str());
            return false;
        }
        n_loaded++;
    }

    fprintf(stderr, "%s: loaded %d tensors (registered %zu)\n",
            __func__, n_loaded, model.tensors.size());

    // Every registered tensor must be present in the file
    if (n_loaded != (int)model.tensors.size()) {
        fprintf(stderr, "%s: tensor count mismatch: file has %d, model registered %zu\n",
                __func__, n_loaded, model.tensors.size());
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model loading — public API
// ═══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<sam3_model> sam3_load_model(const sam3_params& params) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, params.model_path.c_str());

    std::ifstream fin(params.model_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, params.model_path.c_str());
        return nullptr;
    }

    // ── Read + validate header ───────────────────────────────────────────
    uint32_t magic;
    int32_t version, ftype, n_tensors;
    fin.read(reinterpret_cast<char*>(&magic), 4);
    fin.read(reinterpret_cast<char*>(&version), 4);
    fin.read(reinterpret_cast<char*>(&ftype), 4);
    fin.read(reinterpret_cast<char*>(&n_tensors), 4);

    if (magic != SAM3_MAGIC) {
        fprintf(stderr, "%s: invalid magic: 0x%08x (expected 0x%08x)\n",
                __func__, magic, SAM3_MAGIC);
        return nullptr;
    }
    if (version != SAM3_VERSION) {
        fprintf(stderr, "%s: unsupported version: %d (expected %d)\n",
                __func__, version, SAM3_VERSION);
        return nullptr;
    }
    fprintf(stderr, "%s: format version %d, ftype %d, %d tensors\n",
            __func__, version, ftype, n_tensors);

    auto model = std::make_shared<sam3_model>();
    model->weight_type = (ftype == 0) ? GGML_TYPE_F32 : GGML_TYPE_F16;

    // ── Read hyperparameters ─────────────────────────────────────────────
    if (!sam3_load_hparams(fin, model->hparams)) {
        fprintf(stderr, "%s: failed to read hyperparameters\n", __func__);
        return nullptr;
    }
    sam3_print_hparams(model->hparams);

    // ── Init backend ─────────────────────────────────────────────────────
#ifdef GGML_USE_METAL
    if (params.use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model->backend = ggml_backend_metal_init();
    }
#endif
    if (!model->backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model->backend = ggml_backend_cpu_init();
    }
    if (!model->backend) {
        fprintf(stderr, "%s: failed to init backend\n", __func__);
        return nullptr;
    }

    // ── Create ggml context (no_alloc — we use backend_alloc_ctx_tensors)
    // Estimate: ~3000 tensors, generous overhead
    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model->ctx = ggml_init(ctx_params);
    if (!model->ctx) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return nullptr;
    }

    // ── Register all tensor shapes ───────────────────────────────────────
    sam3_register_tensors(*model);
    fprintf(stderr, "%s: registered %zu tensors\n", __func__, model->tensors.size());

    // ── Allocate backend buffer for all tensors ──────────────────────────
    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    if (!model->buffer) {
        fprintf(stderr, "%s: failed to allocate tensor buffer\n", __func__);
        return nullptr;
    }
    fprintf(stderr, "%s: buffer size = %.2f MB\n", __func__,
            ggml_backend_buffer_get_size(model->buffer) / (1024.0 * 1024.0));

    // ── Load tensor data from file ───────────────────────────────────────
    if (!sam3_load_tensors(fin, *model)) {
        fprintf(stderr, "%s: failed to load tensors\n", __func__);
        return nullptr;
    }

    // ── Load BPE tokenizer ───────────────────────────────────────────────
    {
        std::string tok_dir = params.tokenizer_dir;
        if (tok_dir.empty()) {
            // Default: same directory as the model file
            auto slash = params.model_path.find_last_of("/\\");
            tok_dir = (slash != std::string::npos)
                          ? params.model_path.substr(0, slash)
                          : ".";
        }
        if (!sam3_load_bpe_vocab(model->tokenizer, tok_dir)) {
            fprintf(stderr,
                    "%s: WARNING: tokenizer not loaded from '%s' "
                    "(text prompts will not work)\n",
                    __func__, tok_dir.c_str());
        }
    }

    fprintf(stderr, "%s: model loaded successfully\n", __func__);
    return model;
}

void sam3_free_model(sam3_model& model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Inference state
// ═══════════════════════════════════════════════════════════════════════════════

// Deleter implementations for opaque types
void sam3_state_deleter::operator()(sam3_state* p) const {
    if (p) {
        sam3_free_state(*p);
        delete p;
    }
}

void sam3_tracker_deleter::operator()(sam3_tracker* p) const {
    if (p) {
        sam3_tracker_reset(*p);
        delete p;
    }
}

sam3_state_ptr sam3_create_state(const sam3_model& model,
                                 const sam3_params& params) {
    sam3_state_ptr state(new sam3_state());
    state->backend = model.backend;
    state->n_threads = (params.n_threads > 0)
                         ? params.n_threads
                         : std::max(1u, std::thread::hardware_concurrency());
    return state;
}

void sam3_free_state(sam3_state& state) {
    if (state.galloc) {
        ggml_gallocr_free(state.galloc);
        state.galloc = nullptr;
    }
    if (state.buffer) {
        ggml_backend_buffer_free(state.buffer);
        state.buffer = nullptr;
    }
    if (state.pe_buf) {
        ggml_backend_buffer_free(state.pe_buf);
        state.pe_buf = nullptr;
    }
    if (state.pe_ctx) {
        ggml_free(state.pe_ctx);
        state.pe_ctx = nullptr;
    }
    if (state.ctx) {
        ggml_free(state.ctx);
        state.ctx = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image preprocessing
// ═══════════════════════════════════════════════════════════════════════════════

// Bilinear resize of a [H, W, 3] uint8 image to [dst_h, dst_w, 3].
static void sam3_resize_bilinear(const uint8_t* src, int src_w, int src_h,
                                 uint8_t* dst, int dst_w, int dst_h) {
    const float sx = (float)src_w / dst_w;
    const float sy = (float)src_h / dst_h;
    for (int y = 0; y < dst_h; ++y) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int y0 = std::max(0, (int)fy);
        const int y1 = std::min(src_h - 1, y0 + 1);
        const float wy = fy - y0;
        for (int x = 0; x < dst_w; ++x) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int x0 = std::max(0, (int)fx);
            const int x1 = std::min(src_w - 1, x0 + 1);
            const float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                float v = (1 - wy) * ((1 - wx) * src[(y0 * src_w + x0) * 3 + c] +
                                      wx * src[(y0 * src_w + x1) * 3 + c]) +
                          wy * ((1 - wx) * src[(y1 * src_w + x0) * 3 + c] +
                                wx * src[(y1 * src_w + x1) * 3 + c]);
                dst[(y * dst_w + x) * 3 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
}

// Preprocess an image: resize to img_size × img_size, convert to float, normalize.
// Returns a float tensor in [C, H, W] layout (channel-first), range normalized with
// mean=0.5, std=0.5 → pixel values in [-1, 1].
static std::vector<float> sam3_preprocess_image(const sam3_image& image, int img_size) {
    const int C = 3;
    std::vector<float> result(C * img_size * img_size);

    // Resize to img_size × img_size
    std::vector<uint8_t> resized;
    const uint8_t* pixels = image.data.data();
    int w = image.width, h = image.height;

    if (w != img_size || h != img_size) {
        resized.resize(img_size * img_size * 3);
        sam3_resize_bilinear(pixels, w, h, resized.data(), img_size, img_size);
        pixels = resized.data();
        w = img_size;
        h = img_size;
    }

    // Convert to float [C, H, W] with normalization: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                float v = pixels[(y * img_size + x) * 3 + c] / 255.0f;
                result[c * img_size * img_size + y * img_size + x] = (v - 0.5f) / 0.5f;
            }
        }
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RoPE — 2D axial rotary positional embeddings
// ═══════════════════════════════════════════════════════════════════════════════

// Precompute RoPE frequencies as [N, head_dim/2, 2] (cos, sin pairs).
// This matches compute_axial_cis() from vitdet.py stored as real (cos, sin)
// instead of complex numbers.
// The conversion script already stores freqs_cis per block, so this function
// is only needed if we want to recompute them from scratch.
static void sam3_compute_axial_cis(float* out,
                                   int dim, int end_x, int end_y,
                                   float theta, float scale_pos) {
    const int half_dim = dim / 4;  // 16 for dim=64

    // Compute frequency bases: 1.0 / (theta ^ (arange(0,dim,4)[:dim//4] / dim))
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / powf(theta, (float)(i * 4) / (float)dim);
    }

    // For each spatial position, compute axial frequencies
    const int N = end_x * end_y;
    for (int idx = 0; idx < N; ++idx) {
        float t_x = (float)(idx % end_x) * scale_pos;
        int row = idx / end_x;  // intentional integer floor division (row index)
        float t_y = (float)row * scale_pos;

        // X frequencies → first 16 complex values (stored as cos, sin)
        for (int i = 0; i < half_dim; ++i) {
            float angle_x = t_x * freqs[i];
            out[idx * dim + i * 2 + 0] = cosf(angle_x);
            out[idx * dim + i * 2 + 1] = sinf(angle_x);
        }
        // Y frequencies → next 16 complex values
        for (int i = 0; i < half_dim; ++i) {
            float angle_y = t_y * freqs[i];
            out[idx * dim + half_dim * 2 + i * 2 + 0] = cosf(angle_y);
            out[idx * dim + half_dim * 2 + i * 2 + 1] = sinf(angle_y);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sinusoidal 2D positional encoding (for FPN neck outputs)
// ═══════════════════════════════════════════════════════════════════════════════

// Generates sinusoidal PE matching PositionEmbeddingSine from Python.
// num_pos_feats = d_model / 2 = 128, temperature = 10000, normalize = true, scale = 2pi.
// Returns data in ggml column-major layout for a tensor with ne = {d_model, W, H, 1},
// i.e. element (c, w, h) at flat index c + w*d_model + h*d_model*W.
// First half channels (0..half-1) encode y, second half (half..d_model-1) encode x.
static std::vector<float> sam3_sinusoidal_pe_2d(int H, int W, int d_model) {
    const int half = d_model / 2;  // 128
    const float scale = 2.0f * (float)M_PI;
    const float temperature = 10000.0f;

    std::vector<float> pe(d_model * H * W);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Normalized positions: (pos+1) / (max_pos+1) * scale
            float pos_y = ((float)(y + 1) / (float)(H)) * scale;
            float pos_x = ((float)(x + 1) / (float)(W)) * scale;

            for (int i = 0; i < half; ++i) {
                int paired = i & ~1;  // 0,0,2,2,4,4,… (pairs sin/cos channels, matches Python // 2)
                float dim_t = powf(temperature, (float)paired / (float)half);

                float val_x, val_y;
                if (i % 2 == 0) {
                    val_x = sinf(pos_x / dim_t);
                    val_y = sinf(pos_y / dim_t);
                } else {
                    val_x = cosf(pos_x / dim_t);
                    val_y = cosf(pos_y / dim_t);
                }

                // ggml layout: ne = {d_model, W, H, 1}
                // element (c, x, y, 0) at flat index: c + x*d_model + y*d_model*W
                // First half channels are y, second half are x.
                pe[(i) + x * d_model + y * d_model * W] = val_y;
                pe[(i + half) + x * d_model + y * d_model * W] = val_x;
            }
        }
    }

    return pe;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ViT forward pass — graph building
// ═══════════════════════════════════════════════════════════════════════════════

// All ViT graph functions use the sam.cpp convention:
//   ne[0] = embed_dim (E=1024), ne[1] = spatial W, ne[2] = spatial H, ne[3] = batch

// Apply RoPE to Q and K tensors using complex multiplication.
// x shape: [head_dim, N, num_heads*B] in ggml layout
// freqs_cis shape: [2, 32, N] in ggml layout — stored as (cos,sin) interleaved pairs
//
// Python's apply_rotary_enc does:
//   xq_ = view_as_complex(xq.reshape(..., -1, 2))  # pairs consecutive dims
//   xq_out = view_as_real(xq_ * freqs_cis).flatten(3)
//
// In real arithmetic: for each pair (x[2i], x[2i+1]) and freq (cos, sin):
//   out[2i]   = x[2i]*cos - x[2i+1]*sin
//   out[2i+1] = x[2i]*sin + x[2i+1]*cos
static struct ggml_tensor* sam3_apply_rope(struct ggml_context* ctx,
                                           struct ggml_tensor* x,
                                           struct ggml_tensor* freqs_cis) {
    // freqs_cis: [2, 32, N] — dim0=2 (cos,sin), dim1=32 (half_head=head_dim/2), dim2=N
    // x: [head_dim, N, num_heads*B] — dim0=64, dim1=N, dim2=batch*heads

    const int64_t head_dim = x->ne[0];  // 64
    const int64_t N = x->ne[1];         // number of tokens
    const int64_t nheads_B = x->ne[2];  // num_heads * batch
    const int64_t half = head_dim / 2;  // 32

    // Reshape x to [2, half, N, nheads_B] to expose (real, imag) pairs
    auto* x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);

    // freqs_cis: [2, 32, N] → [2, half, N, 1] for broadcast
    auto* fc = ggml_reshape_4d(ctx, freqs_cis, 2, half, N, 1);

    // Extract cos (offset 0) and sin (offset 1) from dim0.
    // fc is [2, half, N, 1] — to slice dim0 we keep strides of dims 1,2,3
    // as nb1,nb2,nb3 of the view, so the view walks over (half, N, 1) correctly.
    auto* cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                               fc->nb[1], fc->nb[2], fc->nb[3], 0);
    auto* sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                               fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);

    // Extract x_re (offset 0) and x_im (offset 1) from dim0.
    // x_pairs is [2, half, N, nheads_B] — same slicing logic.
    auto* x_re = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    auto* x_im = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                              x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);

    // Complex multiply: (x_re + j*x_im) * (cos + j*sin)
    auto* out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f), ggml_mul(ctx, x_im, sin_f));
    auto* out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f), ggml_mul(ctx, x_im, cos_f));

    // Interleave back: [2, half, N, nheads_B]
    auto* out = ggml_concat(ctx, out_re, out_im, 0);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), head_dim, N, nheads_B);
}

// Single ViT block forward: pre-norm → attn (window or global, with RoPE) → residual → pre-norm → MLP → residual
// x: [E, W, H, B] in ggml layout (following sam.cpp convention)
static struct ggml_tensor* sam3_vit_block_forward(struct ggml_context* ctx,
                                                  struct ggml_tensor* x,
                                                  const sam3_vit_block& blk,
                                                  const sam3_hparams& hp,
                                                  int block_idx) {
    const int E = hp.vit_embed_dim;     // 1024
    const int NH = hp.vit_num_heads;    // 16
    const int HD = hp.vit_head_dim();   // 64
    const int WS = hp.vit_window_size;  // 24
    const bool is_global = hp.is_global_attn(block_idx);

    auto* shortcut = x;

    // Pre-norm (normalizes over ne[0] = E)
    x = sam3_layer_norm(ctx, x, blk.norm1_w, blk.norm1_b);

    // Save spatial dims for window unpartition
    const int64_t w0 = x->ne[1];
    const int64_t h0 = x->ne[2];

    if (!is_global) {
        // Window partition: [E, W, H, B] → [E, WS, WS, B*num_windows]
        x = ggml_win_part(ctx, x, WS);
    }

    const int64_t W_cur = x->ne[1];
    const int64_t H_cur = x->ne[2];
    const int64_t B_cur = x->ne[3];

    // ── Self-attention ────────────────────────────────────────────────────
    {
        // QKV projection
        auto* cur = ggml_mul_mat(ctx, blk.qkv_w, x);
        cur = ggml_add(ctx, cur, blk.qkv_b);
        // cur: [3*E, W_cur, H_cur, B_cur]

        // Reshape and permute to separate Q, K, V (following sam.cpp pattern)
        // [3*E, W*H, B_cur] → [E, 3, W*H, B_cur] → permute(0,3,1,2) → [E, W*H, B_cur, 3]
        cur = ggml_reshape_4d(ctx, cur, E, 3, W_cur * H_cur, B_cur);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));
        // cur: [E, W*H, B_cur, 3]  (ne[3]=3 separates Q/K/V)

        auto* Q = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 0);
        auto* K = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
        auto* V = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                               cur->nb[1], cur->nb[2], 2 * cur->nb[3]);

        // Reshape to multi-head: [HD, N, NH*B_cur]
        Q = ggml_reshape_4d(ctx, Q, HD, NH, W_cur * H_cur, B_cur);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx, Q, HD, W_cur * H_cur, NH * B_cur);

        K = ggml_reshape_4d(ctx, K, HD, NH, W_cur * H_cur, B_cur);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx, K, HD, W_cur * H_cur, NH * B_cur);

        V = ggml_reshape_4d(ctx, V, HD, NH, W_cur * H_cur, B_cur);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
        V = ggml_reshape_3d(ctx, V, HD, W_cur * H_cur, NH * B_cur);

        // Apply RoPE to Q and K
        if (blk.freqs_cis) {
            Q = sam3_apply_rope(ctx, Q, blk.freqs_cis);
            K = sam3_apply_rope(ctx, K, blk.freqs_cis);
        }

        // Reshape for flash attention: [HD, N, NH, B_cur]
        Q = ggml_reshape_4d(ctx, Q, HD, W_cur * H_cur, NH, B_cur);
        K = ggml_reshape_4d(ctx, K, HD, W_cur * H_cur, NH, B_cur);
        V = ggml_reshape_4d(ctx, V, HD, W_cur * H_cur, NH, B_cur);

        float scale = 1.0f / sqrtf((float)HD);
        auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // flash_attn_ext result: [HD, NH, N, B_cur]
        // This is already in the right order for head merging:
        // contiguous layout has HD and NH adjacent, so reshape to [E, N, B_cur]
        // then to [E, W, H, B] works correctly.
        x = ggml_reshape_4d(ctx, attn_out, E, W_cur, H_cur, B_cur);

        // Output projection
        x = ggml_mul_mat(ctx, blk.proj_w, x);
        x = ggml_add(ctx, x, blk.proj_b);
    }

    if (!is_global) {
        // Window unpartition
        x = ggml_win_unpart(ctx, x, w0, h0, WS);
    }

    // Residual connection
    x = ggml_add(ctx, shortcut, x);

    // ── FFN ───────────────────────────────────────────────────────────────
    shortcut = x;

    // Pre-norm
    x = sam3_layer_norm(ctx, x, blk.norm2_w, blk.norm2_b);

    // MLP: fc1 → GELU → fc2  (ggml_mul_mat operates on ne[0])
    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    x = ggml_gelu_erf(ctx, x);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    // Residual
    x = ggml_add(ctx, shortcut, x);

    return x;
}

// Build the full ViT graph.
// Input: [img_size, img_size, 3, 1] (ggml convention: [W, H, C, B])
// Output: [E, W, H, 1] where E=1024, W=H=72
static struct ggml_tensor* sam3_build_vit_graph(struct ggml_context* ctx,
                                                struct ggml_tensor* input,
                                                const sam3_model& model) {
    const auto& hp = model.hparams;
    const int E = hp.vit_embed_dim;  // 1024
    const int H = hp.n_img_embd();   // 72
    const int W = hp.n_img_embd();   // 72

    // ── Patch embedding ───────────────────────────────────────────────────
    // Conv2d(3, 1024, k=14, s=14, no bias)
    // Input: [img_size, img_size, 3, 1]
    // Output: [W, H, E, 1]  (ggml conv output convention)
    auto* x = ggml_conv_2d_sk_p0(ctx, model.vit.patch_embed_w, input);

    // Permute to [E, W, H, B] (sam.cpp convention: embed dim first)
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));

    // ── Positional embedding (tiled) ──────────────────────────────────────
    // pos_embed: [E, 24, 24, 1] — Hiera pretrained resolution, no cls token.
    // Tile 3x3 to match [E, 72, 72, 1].
    auto* pos_2d = model.vit.pos_embed;  // [E, 24, 24, 1]

    // Tile 3×3 using ggml_repeat to match [E, 72, 72, 1]
    auto* pos_target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, W, H, 1);
    auto* pos_tiled = ggml_repeat(ctx, pos_2d, pos_target);

    x = ggml_add(ctx, x, pos_tiled);

    // ── LayerNorm pre ─────────────────────────────────────────────────────
    x = sam3_layer_norm(ctx, x, model.vit.ln_pre_w, model.vit.ln_pre_b);

    // ── 32 transformer blocks ─────────────────────────────────────────────
    for (int i = 0; i < hp.vit_depth; ++i) {
        x = sam3_vit_block_forward(ctx, x, model.vit.blocks[i], hp, i);
    }

    // Output: [E, W, H, 1] = [1024, 72, 72, 1]
    return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Neck (SimpleFPN) — graph building
// ═══════════════════════════════════════════════════════════════════════════════

// Build the SimpleFPN neck graph for one path (detector or tracker).
// Input: ViT output [E, W, H, B] with E=1024, W=H=72
// But the conv ops expect [W, H, C, B], so we must permute before convolutions.
// Output: 4 feature maps at different scales in [C, W, H, B] layout.
//   out[0]: [256, 288, 288, B]  (4× upsample)
//   out[1]: [256, 144, 144, B]  (2× upsample)
//   out[2]: [256,  72,  72, B]  (1×)
//   out[3]: [256,  36,  36, B]  (0.5× downsample)
static void sam3_build_neck_graph(struct ggml_context* ctx,
                                  struct ggml_tensor* vit_out,
                                  const sam3_neck& neck,
                                  struct ggml_tensor* out[4]) {
    // Permute from [E, W, H, B] to [W, H, E, B] for conv operations
    auto* x = ggml_cont(ctx, ggml_permute(ctx, vit_out, 2, 0, 1, 3));

    // Helper: add bias to conv output.
    // Conv output is [W, H, C, B]. Bias is [C] (1D).
    // Reshape bias to [1, 1, C, 1] so ggml_repeat can broadcast.
    auto add_bias = [&](struct ggml_tensor* conv_out, struct ggml_tensor* bias) -> struct ggml_tensor* {
        auto* b3d = ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]);
        return ggml_add(ctx, conv_out, ggml_repeat(ctx, b3d, conv_out));
    };

    // Scale 0 (4×): ConvTranspose(1024→512, k=2, s=2) → GELU → ConvTranspose(512→256, k=2, s=2) → Conv1x1 → Conv3x3
    {
        auto* s0 = ggml_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[0].deconv1_w), x, 2);
        s0 = add_bias(s0, neck.scales[0].deconv1_b);
        s0 = ggml_gelu_erf(ctx, s0);
        s0 = ggml_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[0].deconv2_w), s0, 2);
        s0 = add_bias(s0, neck.scales[0].deconv2_b);
        s0 = ggml_conv_2d_sk_p0(ctx, neck.scales[0].conv1x1_w, s0);
        s0 = add_bias(s0, neck.scales[0].conv1x1_b);
        s0 = ggml_conv_2d_s1_ph(ctx, neck.scales[0].conv3x3_w, s0);
        s0 = add_bias(s0, neck.scales[0].conv3x3_b);
        // Permute back to [C, W, H, B]
        out[0] = ggml_cont(ctx, ggml_permute(ctx, s0, 1, 2, 0, 3));
    }

    // Scale 1 (2×): ConvTranspose(1024→512, k=2, s=2) → Conv1x1(512→256) → Conv3x3
    {
        auto* s1 = ggml_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, neck.scales[1].deconv1_w), x, 2);
        s1 = add_bias(s1, neck.scales[1].deconv1_b);
        s1 = ggml_conv_2d_sk_p0(ctx, neck.scales[1].conv1x1_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv1x1_b);
        s1 = ggml_conv_2d_s1_ph(ctx, neck.scales[1].conv3x3_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv3x3_b);
        out[1] = ggml_cont(ctx, ggml_permute(ctx, s1, 1, 2, 0, 3));
    }

    // Scale 2 (1×): Conv1x1(1024→256) → Conv3x3
    {
        auto* s2 = ggml_conv_2d_sk_p0(ctx, neck.scales[2].conv1x1_w, x);
        s2 = add_bias(s2, neck.scales[2].conv1x1_b);
        s2 = ggml_conv_2d_s1_ph(ctx, neck.scales[2].conv3x3_w, s2);
        s2 = add_bias(s2, neck.scales[2].conv3x3_b);
        out[2] = ggml_cont(ctx, ggml_permute(ctx, s2, 1, 2, 0, 3));
    }

    // Scale 3 (0.5×): MaxPool(k=2, s=2) → Conv1x1(1024→256) → Conv3x3
    {
        auto* s3 = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        s3 = ggml_conv_2d_sk_p0(ctx, neck.scales[3].conv1x1_w, s3);
        s3 = add_bias(s3, neck.scales[3].conv1x1_b);
        s3 = ggml_conv_2d_s1_ph(ctx, neck.scales[3].conv3x3_w, s3);
        s3 = add_bias(s3, neck.scales[3].conv3x3_b);
        out[3] = ggml_cont(ctx, ggml_permute(ctx, s3, 1, 2, 0, 3));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Text Encoder — graph building (Phase 4)
// ═══════════════════════════════════════════════════════════════════════════════

// Build a causal (lower-triangular) attention mask for the text encoder.
// Returns: [L, L] F16 tensor. mask[kv][q] = 0 if kv <= q, -inf otherwise.
// Marked as input — caller must upload data via ggml_backend_tensor_set after alloc.
static struct ggml_tensor* sam3_build_causal_mask(struct ggml_context* ctx, int L) {
    auto* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);
    return mask;
}

// Fill a pre-allocated causal mask buffer (host-side, F16).
// mask_data must hold L*L ggml_fp16_t values.
static void sam3_fill_causal_mask(ggml_fp16_t* mask_data, int L) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < L; ++q) {
        for (int kv = 0; kv < L; ++kv) {
            mask_data[kv + q * L] = (kv <= q) ? zero : neginf;
        }
    }
}

// Single text encoder block forward pass.
// Input x: [E, L] where E=text_width=1024, L=seq_len (typically 32).
// causal_mask: [L, L] F16 additive mask for ggml_flash_attn_ext.
// Returns: [E, L]
static struct ggml_tensor* sam3_text_block_forward(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   const sam3_text_block& blk,
                                                   const sam3_hparams& hp,
                                                   struct ggml_tensor* causal_mask,
                                                   int block_idx) {
    const int E = hp.text_width;   // 1024
    const int NH = hp.text_heads;  // 16
    const int HD = E / NH;         // 64
    const int64_t L = x->ne[1];    // sequence length

    // ── Self-attention with causal mask ──────────────────────────────────
    auto* shortcut = x;

    // Pre-norm
    x = sam3_layer_norm(ctx, x, blk.ln1_w, blk.ln1_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln1", block_idx);

    // QKV projection: [E, L] → [3*E, L]
    auto* qkv = ggml_mul_mat(ctx, blk.attn_in_proj_w, x);
    qkv = ggml_add(ctx, qkv, blk.attn_in_proj_b);
    sam3_name_tensorf(qkv, "text_block_%02d_qkv", block_idx);

    // Split Q, K, V: reshape [3*E, L] → [E, 3, L] → permute → [E, L, 3]
    qkv = ggml_reshape_3d(ctx, qkv, E, 3, L);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 2, 1, 3));
    // qkv: [E, L, 3]

    auto* Q = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 0);
    auto* K = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 1 * qkv->nb[2]);
    auto* V = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 2 * qkv->nb[2]);

    // Reshape for multi-head flash attention:
    //   [E, L] → [HD, NH, L] → permute(0,2,1) → [HD, L, NH] → [HD, L, NH, 1]
    Q = ggml_reshape_3d(ctx, Q, HD, NH, L);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_4d(ctx, Q, HD, L, NH, 1);

    K = ggml_reshape_3d(ctx, K, HD, NH, L);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_4d(ctx, K, HD, L, NH, 1);

    V = ggml_reshape_3d(ctx, V, HD, NH, L);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
    V = ggml_reshape_4d(ctx, V, HD, L, NH, 1);

    // flash_attn_ext:
    //   Q: [HD, L, NH, 1], K: [HD, L, NH, 1], V: [HD, L, NH, 1]
    //   mask: [L, L] (causal)
    //   result: [HD, NH, L, 1]
    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, causal_mask, scale, 0.0f, 0.0f);

    // Reshape [HD, NH, L, 1] → [E, L]
    x = ggml_reshape_2d(ctx, attn_out, E, L);

    // Output projection
    x = ggml_mul_mat(ctx, blk.attn_out_proj_w, x);
    x = ggml_add(ctx, x, blk.attn_out_proj_b);

    // LayerScale (if present)
    if (blk.ls1) {
        x = ggml_mul(ctx, x, blk.ls1);
    }
    sam3_name_tensorf(x, "text_block_%02d_attn_out", block_idx);

    // Residual
    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_after_attn_residual", block_idx);

    // ── MLP ─────────────────────────────────────────────────────────────
    shortcut = x;

    // Pre-norm
    x = sam3_layer_norm(ctx, x, blk.ln2_w, blk.ln2_b);
    sam3_name_tensorf(x, "text_block_%02d_after_ln2", block_idx);

    // MLP: fc1(1024→4096) → GELU → fc2(4096→1024)
    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    sam3_name_tensorf(x, "text_block_%02d_mlp_fc1", block_idx);
    x = ggml_gelu_erf(ctx, x);
    sam3_name_tensorf(x, "text_block_%02d_mlp_gelu", block_idx);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    // LayerScale (if present)
    if (blk.ls2) {
        x = ggml_mul(ctx, x, blk.ls2);
    }
    sam3_name_tensorf(x, "text_block_%02d_mlp_out", block_idx);

    // Residual
    x = ggml_add(ctx, shortcut, x);
    sam3_name_tensorf(x, "text_block_%02d_out", block_idx);

    return x;
}

// Build the full text encoder computation graph.
// token_ids: [L] int32 tensor (BPE token IDs, padded to ctx_len with 0s).
//            Must be marked as input by caller; data uploaded after alloc.
// Returns: text_features tensor [text_out_dim, L] = [256, L].
// Also creates the causal mask internally (marked as input).
static struct ggml_tensor* sam3_build_text_encoder_graph(struct ggml_context* ctx,
                                                         struct ggml_tensor* token_ids,
                                                         const sam3_model& model) {
    const auto& hp = model.hparams;
    const auto& enc = model.text_enc;
    const int L = hp.text_ctx_len;  // 32

    // ── Token embedding: lookup [vocab, E] → [E, L] ─────────────────────
    auto* x = ggml_get_rows(ctx, enc.token_embed_w, token_ids);
    // x: [E, L] = [1024, 32]
    ggml_set_name(x, "text_token_embed");

    // ── Add positional embedding ─────────────────────────────────────────
    // pos_embed: [E, ctx_len] = [1024, 32]
    x = ggml_add(ctx, x, enc.pos_embed);
    ggml_set_name(x, "text_after_pos_embed");

    // ── Build causal mask ────────────────────────────────────────────────
    auto* causal_mask = sam3_build_causal_mask(ctx, L);

    // ── 24 transformer blocks ────────────────────────────────────────────
    for (int i = 0; i < hp.text_layers; ++i) {
        x = sam3_text_block_forward(ctx, x, enc.blocks[i], hp, causal_mask, i);
    }

    // ── Final LayerNorm ──────────────────────────────────────────────────
    x = sam3_layer_norm(ctx, x, enc.ln_final_w, enc.ln_final_b);
    ggml_set_name(x, "text_final_ln");

    // ── Resizer projection: Linear(1024 → 256) ──────────────────────────
    x = ggml_mul_mat(ctx, enc.resizer_w, x);
    x = ggml_add(ctx, x, enc.resizer_b);
    ggml_set_name(x, "text_features_2d");
    // x: [OD, L] = [256, 32]

    return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image backbone — public API
// ═══════════════════════════════════════════════════════════════════════════════

bool sam3_encode_image(sam3_state& state,
                       const sam3_model& model,
                       const sam3_image& image) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto& hp = model.hparams;
    const int img_size = hp.img_size;

    fprintf(stderr, "%s: encoding %dx%d image → %dx%d\n", __func__,
            image.width, image.height, img_size, img_size);

    // Save original dimensions
    state.orig_width = image.width;
    state.orig_height = image.height;

    // ── Preprocess image ──────────────────────────────────────────────────
    auto img_data = sam3_preprocess_image(image, img_size);

    // ── Build computation graph ───────────────────────────────────────────
    // Create a temporary ggml context for graph building (no data, just ops)
    // We need enough memory for all intermediate tensors during graph construction.
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    // Create input tensor
    auto* inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_size, img_size, 3, 1);
    ggml_set_name(inp, "input_image");
    ggml_set_input(inp);

    // Build ViT graph
    auto* vit_out = sam3_build_vit_graph(ctx0, inp, model);
    ggml_set_name(vit_out, "vit_output");
    ggml_set_output(vit_out);

    // Build neck graphs (detector and tracker paths)
    struct ggml_tensor* neck_det_out[4];
    struct ggml_tensor* neck_trk_out[4];
    sam3_build_neck_graph(ctx0, vit_out, model.neck_det, neck_det_out);
    sam3_build_neck_graph(ctx0, vit_out, model.neck_trk, neck_trk_out);

    for (int i = 0; i < 4; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "neck_det_%d", i);
        ggml_set_name(neck_det_out[i], name);
        ggml_set_output(neck_det_out[i]);
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        ggml_set_name(neck_trk_out[i], name);
        ggml_set_output(neck_trk_out[i]);
    }

    // Build computation graph
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 16384, false);
    for (int i = 0; i < 4; ++i) {
        ggml_build_forward_expand(graph, neck_det_out[i]);
        ggml_build_forward_expand(graph, neck_trk_out[i]);
    }

    // ── Allocate and compute ──────────────────────────────────────────────
    // Create graph allocator
    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    // Reserve memory (measure pass)
    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    // Allocate tensors
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    fprintf(stderr, "%s: graph allocated, %d nodes\n", __func__, ggml_graph_n_nodes(graph));

    // ggml tensor [W=img_size, H=img_size, C=3, B=1] layout matches our CHW data:
    // ggml offset for (x,y,c,0) = x + y*W + c*W*H = same as CHW[c*H*W + y*W + x]
    // since W=H=img_size. So we can copy directly.
    ggml_backend_tensor_set(inp, img_data.data(), 0, img_data.size() * sizeof(float));

    // Compute
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        sam3_graph_compute(model.backend, graph, state.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "%s: graph computed in %.1f ms (%d threads)\n",
                __func__, ms, state.n_threads);
    }

    // ── Cache results in state ────────────────────────────────────────────
    // TODO: copy output tensors to state for later use by PCS/PVS/tracker
    // For now, store the graph allocator so tensors stay alive
    if (state.galloc) ggml_gallocr_free(state.galloc);
    if (state.ctx) ggml_free(state.ctx);

    state.ctx = ctx0;
    state.galloc = galloc;
    state.backend = model.backend;
    state.vit_output = vit_out;

    // Store neck outputs
    for (int i = 0; i < 4; ++i) {
        state.neck_det[i] = neck_det_out[i];
        state.neck_trk[i] = neck_trk_out[i];
    }

    // Compute sinusoidal PEs for each neck scale.
    // Neck output layout is [C, W, H, B] where C=neck_dim=256.
    // Allocate a separate ggml context + backend buffer for PEs so they
    // survive independently of the graph allocator.
    {
        const int neck_dim = hp.neck_dim;  // 256
        const int scale_sizes[4] = {
            hp.n_img_embd() * 4,  // 288
            hp.n_img_embd() * 2,  // 144
            hp.n_img_embd(),      //  72
            hp.n_img_embd() / 2,  //  36
        };

        // Compute total bytes needed for all 4 PE tensors
        size_t pe_total = 0;
        for (int i = 0; i < 4; ++i) {
            pe_total += (size_t)neck_dim * scale_sizes[i] * scale_sizes[i] * sizeof(float);
        }

        // Free previous PE resources if re-encoding
        if (state.pe_buf) {
            ggml_backend_buffer_free(state.pe_buf);
            state.pe_buf = nullptr;
        }
        if (state.pe_ctx) {
            ggml_free(state.pe_ctx);
            state.pe_ctx = nullptr;
        }

        // Create a ggml context for PE tensor metadata (4 tensors + overhead)
        struct ggml_init_params pe_params = {
            /*.mem_size   =*/ggml_tensor_overhead() * 4 + 256,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        state.pe_ctx = ggml_init(pe_params);

        // Create tensor descriptors
        struct ggml_tensor* pe_tensors[4];
        for (int i = 0; i < 4; ++i) {
            const int S = scale_sizes[i];
            pe_tensors[i] = ggml_new_tensor_4d(state.pe_ctx, GGML_TYPE_F32, neck_dim, S, S, 1);
            char name[64];
            snprintf(name, sizeof(name), "pe_%d", i);
            ggml_set_name(pe_tensors[i], name);
        }

        // Allocate a single backend buffer and assign tensors
        state.pe_buf = ggml_backend_alloc_ctx_tensors(state.pe_ctx, model.backend);
        if (!state.pe_buf) {
            fprintf(stderr, "%s: failed to allocate PE buffer\n", __func__);
        } else {
            // Upload PE data
            for (int i = 0; i < 4; ++i) {
                const int S = scale_sizes[i];
                auto pe_data = sam3_sinusoidal_pe_2d(S, S, neck_dim);
                ggml_backend_tensor_set(pe_tensors[i], pe_data.data(), 0, pe_data.size() * sizeof(float));

                state.neck_det_pe[i] = pe_tensors[i];
                // Tracker shares the same spatial dimensions → same PE
                state.neck_trk_pe[i] = pe_tensors[i];
            }
        }
    }

    // Invalidate PE cache so it's re-populated on next PVS call if needed
    state.pe_cache_valid = false;

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stderr, "%s: image encoded successfully in %.1f ms\n", __func__, total_ms);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Multi-head attention helper (used by fusion encoder, DETR decoder, seg head)
// ═══════════════════════════════════════════════════════════════════════════════

// Standard multi-head attention with fused in_proj.
// q_in, k_in, v_in: [D, N, B]  (if fused_qkv, only q_in is used and contains QKV stacked)
// in_proj_w: [D, 3*D] (fused Q/K/V projection)
// in_proj_b: [3*D]
// out_proj_w: [D, D], out_proj_b: [D]
// n_heads: number of attention heads
// Returns: [D, N_q, B]
//
// If separate_kv is true, q_in/k_in/v_in are already separate (no fused proj needed).
// The in_proj is applied to form Q from q_in, and K/V from the concatenated k/v source.
static struct ggml_tensor* sam3_multihead_attn_fused(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,        // [D, N_q, B]
    struct ggml_tensor* kv_in,       // [D, N_kv, B] (can be same as q_in for self-attn)
    struct ggml_tensor* in_proj_w,   // [D, 3*D] — fused QKV weights
    struct ggml_tensor* in_proj_b,   // [3*D]
    struct ggml_tensor* out_proj_w,  // [D, D]
    struct ggml_tensor* out_proj_b,  // [D]
    int n_heads,
    struct ggml_tensor* attn_mask = nullptr)  // [N_kv, N_q] or nullptr
{
    const int64_t D = q_in->ne[0];  // 256
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = kv_in->ne[1];
    const int64_t HD = D / n_heads;

    // Split in_proj into Q, K, V projections via views
    // in_proj_w: [D, 3*D] → q_w=[D, D] (rows 0..D-1), k_w=[D, D] (rows D..2D-1), v_w=[D, D] (rows 2D..3D-1)
    auto* q_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 0);
    auto* k_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], D * in_proj_w->nb[1]);
    auto* v_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 2 * D * in_proj_w->nb[1]);

    auto* q_b = ggml_view_1d(ctx, in_proj_b, D, 0);
    auto* k_b = ggml_view_1d(ctx, in_proj_b, D, D * sizeof(float));
    auto* v_b = ggml_view_1d(ctx, in_proj_b, D, 2 * D * sizeof(float));

    // Project: Q from q_in, K and V from kv_in
    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);   // [D, N_q, B]
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, kv_in), k_b);  // [D, N_kv, B]
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, kv_in), v_b);  // [D, N_kv, B]

    // Reshape to multi-head: [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    // Attention
    float scale = 1.0f / sqrtf((float)HD);
    auto* attn_out = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);
    // Result: [HD, NH, N_q, B] — head dim and n_heads adjacent

    // Merge heads: [D, N_q, B]
    auto* merged = ggml_reshape_3d(ctx, attn_out, D, N_q, B);

    // Output projection
    merged = ggml_mul_mat(ctx, out_proj_w, merged);
    merged = ggml_add(ctx, merged, out_proj_b);

    return merged;
}

static struct ggml_tensor* sam3_expand_token_attn_bias(
    struct ggml_context * ctx,
    struct ggml_tensor * token_bias,  // [T, 1, B] or nullptr
    int64_t n_q,
    int n_heads,
    int64_t batch) {
    if (!token_bias) {
        return nullptr;
    }

    const int64_t n_kv = token_bias->ne[0];
    auto * bias_4d = ggml_reshape_4d(ctx, token_bias, n_kv, 1, 1, batch);
    auto * full_bias = ggml_repeat(
        ctx,
        bias_4d,
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_q, n_heads, batch));

    // ggml_flash_attn_ext expects mask storage in fp16.
    return ggml_cont(ctx, ggml_cast(ctx, full_bias, GGML_TYPE_F16));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Fusion encoder — graph building (6 layers)
// ═══════════════════════════════════════════════════════════════════════════════

// Single fusion encoder layer.
// x: [D, N, B] image features (N=5184), prompt: [D, T, B] text/exemplar tokens, pos: [D, N, B]
// Returns: updated x [D, N, B]
static struct ggml_tensor* sam3_fenc_layer_forward(
    struct ggml_context* ctx,
    const sam3_fenc_layer& ly,
    struct ggml_tensor* x,
    struct ggml_tensor* prompt,
    struct ggml_tensor* pos,
    struct ggml_tensor* prompt_attn_bias,
    int n_heads) {
    // 1. Self-attention on image features (pre-norm)
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
        // q = k = norm(x) + pos, v = norm(x)
        auto* q_in = ggml_add(ctx, x_norm, pos);
        auto* k_in = ggml_add(ctx, x_norm, pos);

        // Self-attention: Q and K have pos, V does not
        // Use fused in_proj but with separate q and kv inputs
        // Since SA uses same source for Q,K,V but Q/K get pos added,
        // we need a custom approach: project q_in for Q, k_in for K, x_norm for V
        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, x_norm), v_b);

        const int64_t N = x->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto* sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);

        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        x = ggml_add(ctx, shortcut, sa_out);
    }

    // 2. Cross-attention (image → prompt tokens) with pre-norm
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
        // ca_q_w is actually a fused in_proj for Q,K,V but Q comes from image, K/V from prompt
        // The registration code stores fused [D, 3*D] weights in ca_q_w (same as fenc sa)
        // Q from x_norm, K/V from prompt
        const int64_t D = x->ne[0];

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);

        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, x_norm), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, prompt), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, prompt), v_b);

        const int64_t N_q = x->ne[1];
        const int64_t N_kv = prompt->ne[1];
        const int64_t B = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        auto * ca_mask = sam3_expand_token_attn_bias(ctx, prompt_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = ggml_flash_attn_ext(ctx, Q, K, V, ca_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);

        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        x = ggml_add(ctx, shortcut, ca_out);
    }

    // 3. FFN (pre-norm, ReLU activation)
    {
        auto* shortcut = x;
        auto* x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);

        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        x = ggml_add(ctx, shortcut, ffn);
    }

    return x;
}

// Build full fusion encoder graph (6 layers).
// image_feats: [D, N, B] where N=5184 (72*72), D=256
// prompt_tokens: [D, T, B] text/exemplar features
// pos_enc: [D, N, B] sinusoidal positional encoding for image features
// Returns: conditioned_features [D, N, B]
static struct ggml_tensor* sam3_build_fenc_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,
    struct ggml_tensor* prompt_tokens,
    struct ggml_tensor* pos_enc,
    struct ggml_tensor* prompt_attn_bias = nullptr) {
    const auto& hp = model.hparams;
    auto* x = image_feats;

    for (int i = 0; i < hp.fenc_layers; ++i) {
        x = sam3_fenc_layer_forward(ctx, model.fenc.layers[i], x, prompt_tokens,
                                    pos_enc, prompt_attn_bias, hp.fenc_heads);
        sam3_name_tensorf(x, "fenc_layer%d_out", i);
    }

    return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DETR decoder — graph building (6 layers)
// ═══════════════════════════════════════════════════════════════════════════════

// inverse_sigmoid: log(x / (1 - x)), clamped to avoid inf
// Python reference uses eps=1e-3: x1 = x.clamp(min=eps), x2 = (1-x).clamp(min=eps)
static struct ggml_tensor* sam3_inverse_sigmoid(struct ggml_context* ctx, struct ggml_tensor* x) {
    // clamp x to [1e-3, 1-1e-3] to match Python eps=1e-3
    x = ggml_clamp(ctx, x, 1e-3f, 1.0f - 1e-3f);
    // log(x / (1 - x)) = log(x) - log(1 - x)
    auto* log_x = ggml_log(ctx, x);
    // Compute (1 - x) as (-1)*x + 1.  We use ggml_scale_bias which takes float
    // scalars (no tensor allocation needed, safe in no_alloc contexts).
    auto* one_minus = ggml_scale_bias(ctx, x, -1.0f, 1.0f);
    auto* log_1mx = ggml_log(ctx, one_minus);
    return ggml_sub(ctx, log_x, log_1mx);
}

// Box refinement MLP (3 layers: D→D→D→4 with ReLU)
static struct ggml_tensor* sam3_bbox_mlp(struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* w[3],
                                         struct ggml_tensor* b[3]) {
    for (int j = 0; j < 3; ++j) {
        x = ggml_mul_mat(ctx, w[j], x);
        x = ggml_add(ctx, x, b[j]);
        if (j < 2) x = ggml_relu(ctx, x);
    }
    return x;
}

// Build sinusoidal positional embedding for 4D reference points in the ggml graph.
// ref_boxes: [4, NQ, B] — (cx, cy, w, h) after sigmoid, B=1
// sine_dim_t: [1, 64] — pre-computed angle multipliers (2π / 10000^(2i/128))
// Returns: [512, NQ, B] sinusoidal embedding matching Python gen_sineembed_for_position
static struct ggml_tensor* sam3_build_sine_pos_embed_4d(
    struct ggml_context* ctx,
    struct ggml_tensor* ref_boxes,     // [4, NQ, B]
    struct ggml_tensor* sine_dim_t) {  // [1, 64]
    const int64_t NQ = ref_boxes->ne[1];

    // Python output order: [cy, cx, w, h] → coord indices from boxes [cx(0),cy(1),w(2),h(3)]
    const int coord_order[4] = {1, 0, 2, 3};

    struct ggml_tensor* coord_embeds[4];

    for (int c = 0; c < 4; ++c) {
        int ci = coord_order[c];
        // Extract one coordinate: view into ref_boxes [4, NQ, 1] at element ci
        auto* coord = ggml_view_2d(ctx, ref_boxes, 1, NQ,
                                    ref_boxes->nb[1], ci * sizeof(float));  // [1, NQ]

        // Outer product: angles[i, q] = dim_t[i] * coord[q]
        // ggml_mul_mat(A=[1,64], B=[1,NQ]) = A^T @ B = [64,1]@[1,NQ] = [64, NQ]
        auto* angles = ggml_mul_mat(ctx, sine_dim_t, coord);  // [64, NQ]

        auto* sin_vals = ggml_sin(ctx, angles);  // [64, NQ]
        auto* cos_vals = ggml_cos(ctx, angles);  // [64, NQ]

        // Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
        auto* sin_r = ggml_reshape_3d(ctx, sin_vals, 1, 64, NQ);
        auto* cos_r = ggml_reshape_3d(ctx, cos_vals, 1, 64, NQ);
        auto* interleaved = ggml_concat(ctx, sin_r, cos_r, 0);  // [2, 64, NQ]
        coord_embeds[c] = ggml_reshape_2d(ctx, interleaved, 128, NQ);
    }

    // Concatenate all 4 coordinates → [512, NQ]
    auto* embed = ggml_concat(ctx, coord_embeds[0], coord_embeds[1], 0);  // [256, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[2], 0);                  // [384, NQ]
    embed = ggml_concat(ctx, embed, coord_embeds[3], 0);                  // [512, NQ]

    return embed;
}

// Build query positional encoding from reference boxes via sine embed + ref_point_head MLP.
// ref_boxes: [4, NQ, 1] — after sigmoid
// sine_dim_t: [1, 64]
// Returns: [D, NQ+1, 1] (zeros for presence token at index 0, MLP output for object queries)
static struct ggml_tensor* sam3_build_query_pos(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,     // [4, NQ, 1]
    struct ggml_tensor* sine_dim_t,    // [1, 64]
    int layer_idx = -1) {
    const auto& tensors = model.tensors;
    const int64_t NQ = ref_boxes->ne[1];
    const int D = model.hparams.neck_dim;  // 256

    // 1. Sine positional embedding: [512, NQ]
    auto* sine_embed = sam3_build_sine_pos_embed_4d(ctx, ref_boxes, sine_dim_t);
    if (layer_idx == 0) {
        ggml_set_name(sine_embed, "ddec_query_sine_0");
    }

    // 2. ref_point_head MLP: 512 → 256 → 256
    // Layer 0: relu(W0 @ sine_embed + b0)
    auto* h = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.0.weight"), sine_embed);
    h = ggml_add(ctx, h, tensors.at("ddec.ref_point_head.layers.0.bias"));
    h = ggml_relu(ctx, h);
    // Layer 1: W1 @ h + b1 (no activation)
    auto* qpos_obj = ggml_mul_mat(ctx, tensors.at("ddec.ref_point_head.layers.1.weight"), h);
    qpos_obj = ggml_add(ctx, qpos_obj, tensors.at("ddec.ref_point_head.layers.1.bias"));
    // qpos_obj: [D, NQ]
    if (layer_idx == 0) {
        ggml_set_name(qpos_obj, "ddec_query_pos_0");
    }

    // 3. Reshape to 3D and prepend zeros for presence token
    qpos_obj = ggml_reshape_3d(ctx, qpos_obj, D, NQ, 1);
    auto* qpos_pres = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    ggml_set_name(qpos_pres, "ddec_query_pos_pres");
    ggml_set_input(qpos_pres);  // zeros — set by caller

    return ggml_concat(ctx, qpos_pres, qpos_obj, 1);  // [D, NQ+1, 1]
}

// Build box-relative positional bias for DETR cross-attention.
// ref_boxes: [4, N_q, B] — (cx, cy, w, h) in [0,1]
// rpb_coords: [feat_hw] — normalized coords [0/H, 1/H, ..., (H-1)/H] (input tensor)
// Returns: bias tensor [N_kv, N_q+1, n_heads, B] for ggml_flash_attn_ext mask
//
// Python _get_rpb_matrix with boxRPB="log":
//   1. boxes → xyxy
//   2. deltas_x[q,w,:2] = [coord_w - x0, coord_w - x1]
//   3. deltas_y[q,h,:2] = [coord_h - y0, coord_h - y1]
//   4. log transform: sign(d*8) * log2(|d*8|+1) / log2(8)
//   5. MLP: [2] → [256] → [n_heads]
//   6. outer sum: B[h,w] = delta_y[h] + delta_x[w]
static struct ggml_tensor* sam3_compute_box_rpb(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* ref_boxes,   // [4, N_q, B]
    struct ggml_tensor* rpb_coords,  // [feat_hw] — pre-filled grid coordinates
    int feat_hw,
    int layer_idx = -1) {
    const int64_t NQ = ref_boxes->ne[1];
    const int NH = model.hparams.ddec_heads;  // 8
    const int W = feat_hw;
    const int H = feat_hw;
    const auto& tensors = model.tensors;

    // ── 1. Convert cxcywh → xyxy ─────────────────────────────────────────
    // ggml_view_2d on strided data is non-contiguous — ggml_scale requires contiguous.
    // Use ggml_cont to make each coordinate slice contiguous.
    auto* cx = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 0));
    auto* cy = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 1 * sizeof(float)));
    auto* bw = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 2 * sizeof(float)));
    auto* bh = ggml_cont(ctx, ggml_view_2d(ctx, ref_boxes, 1, NQ, ref_boxes->nb[1], 3 * sizeof(float)));
    // x0 = cx - w/2, x1 = cx + w/2
    auto* half_w = ggml_scale(ctx, bw, 0.5f);
    auto* half_h = ggml_scale(ctx, bh, 0.5f);
    auto* x0 = ggml_sub(ctx, cx, half_w);  // [1, NQ]
    auto* x1 = ggml_add(ctx, cx, half_w);
    auto* y0 = ggml_sub(ctx, cy, half_h);
    auto* y1 = ggml_add(ctx, cy, half_h);

    // ── 2. Compute deltas via outer subtract ──────────────────────────────
    // coords: [W] → reshape to [W, 1] for outer subtract
    auto* cw = ggml_reshape_2d(ctx, rpb_coords, W, 1);  // [W, 1]

    // Outer subtract: delta[w, q] = coord[w] - edge[q]
    // Use ggml_mul_mat trick: not applicable for subtraction.
    // Instead: repeat coords to [W, NQ], repeat edge to [W, NQ], subtract.
    auto* shape_wn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, NQ);

    auto* cw_rep = ggml_repeat(ctx, cw, shape_wn);      // [W, NQ] (each column = coords)
    auto* x0_t = ggml_cont(ctx, ggml_transpose(ctx, x0));  // [NQ, 1]
    auto* x0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x0_t, 1, NQ), shape_wn);
    auto* x1_t = ggml_cont(ctx, ggml_transpose(ctx, x1));
    auto* x1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, x1_t, 1, NQ), shape_wn);
    auto* y0_t = ggml_cont(ctx, ggml_transpose(ctx, y0));
    auto* y0_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y0_t, 1, NQ), shape_wn);
    auto* y1_t = ggml_cont(ctx, ggml_transpose(ctx, y1));
    auto* y1_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, y1_t, 1, NQ), shape_wn);

    auto* dx0 = ggml_sub(ctx, cw_rep, x0_rep);  // [W, NQ]
    auto* dx1 = ggml_sub(ctx, cw_rep, x1_rep);
    auto* dy0 = ggml_sub(ctx, cw_rep, y0_rep);  // reusing coords for H (H==W)
    auto* dy1 = ggml_sub(ctx, cw_rep, y1_rep);

    // Stack into [2, W, NQ]: reshape each to [1, W, NQ], concat dim 0
    auto* dx0_r = ggml_reshape_3d(ctx, dx0, 1, W, NQ);
    auto* dx1_r = ggml_reshape_3d(ctx, dx1, 1, W, NQ);
    auto* deltas_x = ggml_concat(ctx, dx0_r, dx1_r, 0);  // [2, W, NQ]
    auto* dy0_r = ggml_reshape_3d(ctx, dy0, 1, H, NQ);
    auto* dy1_r = ggml_reshape_3d(ctx, dy1, 1, H, NQ);
    auto* deltas_y = ggml_concat(ctx, dy0_r, dy1_r, 0);  // [2, H, NQ]

    // ── 3. Log transform: sign(d*8) * log2(|d*8|+1) / log2(8) ────────────
    const float scale8 = 8.0f;
    const float inv_log2_8 = 1.0f / log2f(8.0f);  // = 1/3

    auto rpb_log = [&](struct ggml_tensor* d) -> struct ggml_tensor* {
        auto* d8 = ggml_scale(ctx, d, scale8);
        auto* sign_d = ggml_sgn(ctx, d8);
        auto* abs_d = ggml_abs(ctx, d8);
        auto* log_val = ggml_log(ctx, ggml_scale_bias(ctx, abs_d, 1.0f, 1.0f));
        // log2(x) = ln(x) / ln(2)
        log_val = ggml_scale(ctx, log_val, 1.0f / logf(2.0f));
        return ggml_mul(ctx, sign_d, ggml_scale(ctx, log_val, inv_log2_8));
    };

    deltas_x = rpb_log(deltas_x);  // [2, W, NQ]
    deltas_y = rpb_log(deltas_y);  // [2, H, NQ]

    // ── 4. MLP: [2, W*NQ] → [NH, W*NQ] ───────────────────────────────────
    // boxRPB_embed_x: MLP(2, 256, 8, 2) = Linear(2→256)+ReLU, Linear(256→8)
    // Reshape to [2, W*NQ] so matmul treats each (w, q) pair as a sample
    auto rpb_mlp = [&](struct ggml_tensor* d, const char* axis) -> struct ggml_tensor* {
        int64_t spatial = d->ne[1];
        int64_t nq = d->ne[2];
        auto* flat = ggml_reshape_2d(ctx, d, 2, spatial * nq);  // [2, W*NQ]
        auto wn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.weight";
        auto bn0 = std::string("ddec.boxRPB_embed_") + axis + ".layers.0.bias";
        auto wn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.weight";
        auto bn1 = std::string("ddec.boxRPB_embed_") + axis + ".layers.1.bias";
        flat = ggml_mul_mat(ctx, tensors.at(wn0), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn0));
        flat = ggml_relu(ctx, flat);
        flat = ggml_mul_mat(ctx, tensors.at(wn1), flat);
        flat = ggml_add(ctx, flat, tensors.at(bn1));
        // flat: [NH, W*NQ] → reshape to [NH, spatial, NQ]
        return ggml_reshape_3d(ctx, flat, NH, spatial, nq);
    };

    auto* rpb_x = rpb_mlp(deltas_x, "x");  // [NH, W, NQ]
    auto* rpb_y = rpb_mlp(deltas_y, "y");  // [NH, H, NQ]

    // ── 5. Outer sum: B[nh, w, h, q] = rpb_y[nh, h, q] + rpb_x[nh, w, q] ─
    // Reshape for broadcasting:
    //   rpb_y → [NH, 1, H, NQ]
    //   rpb_x → [NH, W, 1, NQ]
    //
    // Keep W in ne[1] so reshaping [NH, W, H, NQ] → [NH, H*W, NQ, 1]
    // preserves Python's flatten(H, W) order where W is the fast spatial axis.
    auto* rpb_y_4d = ggml_reshape_4d(ctx, rpb_y, NH, 1, H, NQ);
    auto* rpb_x_4d = ggml_reshape_4d(ctx, rpb_x, NH, W, 1, NQ);

    // ggml_add broadcasts: where one dim is 1, the other is used
    auto* rpb_hw = ggml_repeat(ctx, rpb_y_4d,
                                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb_hw_x = ggml_repeat(ctx, rpb_x_4d,
                                  ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, W, H, NQ));
    auto* rpb = ggml_add(ctx, rpb_hw, rpb_hw_x);  // [NH, W, H, NQ]

    // ── 6. Reshape to [H*W, NQ, NH, 1] for flash_attn_ext mask ───────────
    // Current: [NH, W, H, NQ]. Need: [N_kv=H*W, NQ, NH, B=1]
    rpb = ggml_reshape_4d(ctx, rpb, NH, H * W, NQ, 1);
    rpb = ggml_cont(ctx, ggml_permute(ctx, rpb, 2, 0, 1, 3));  // [H*W, NQ, NH, 1]

    // Prepend zeros for presence token: mask for presence token has no box-relative bias
    auto* pres_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H * W, 1, NH, 1);
    ggml_set_name(pres_mask, "rpb_pres_zeros");
    ggml_set_input(pres_mask);  // zeros — set by caller

    // [H*W, NQ+1, NH, 1]
    auto * full_rpb = ggml_concat(ctx, pres_mask, rpb, 1);
    full_rpb = ggml_cont(ctx, full_rpb);
    if (layer_idx == 0) {
        ggml_set_name(full_rpb, "ddec_rpb_mask_0");
    }

    return full_rpb;
}

// Single DETR decoder layer.
// queries: [D, N_q, B] where N_q = 201 (200 object queries + 1 presence token)
// query_pos: [D, N_q, B] positional encoding for queries
// enc_feats: [D, N_kv, B] conditioned image features from fusion encoder
// enc_pos: [D, N_kv, B] positional encoding for image features
// text_feats: [D, T, B] text features
// rpb_mask: [N_kv, N_q, n_heads, B] box-relative positional bias (or nullptr)
// Returns: updated queries [D, N_q, B]
static struct ggml_tensor* sam3_ddec_layer_forward(
    struct ggml_context* ctx,
    const sam3_ddec_layer& ly,
    struct ggml_tensor* queries,
    struct ggml_tensor* query_pos,
    struct ggml_tensor* enc_feats,
    struct ggml_tensor* enc_pos,
    struct ggml_tensor* text_feats,
    int n_heads,
    struct ggml_tensor* text_attn_bias = nullptr,
    struct ggml_tensor* rpb_mask = nullptr,
    int layer_idx = -1) {
    const int64_t D = queries->ne[0];

    // Python decoder layer order (all post-norm):
    //   1. Self-attention → norm2 (post-norm)
    //   2. Text cross-attention (ca_text) → catext_norm (post-norm)
    //   3. Image cross-attention (cross_attn) → norm1 (post-norm)
    //   4. FFN → norm3 (post-norm)
    //
    // Norm weight mapping:
    //   ly.norm2_w  = ".norm2.weight"        = Python norm2 (post-SA)
    //   ly.norm3_w  = ".norm_ca_text.weight"  = Python catext_norm (post-text-CA)
    //   ly.norm1_w  = ".norm1.weight"         = Python norm1 (post-image-CA)
    //   ly.norm4_w  = ".norm3.weight"         = Python norm3 (post-FFN)

    // 1. Self-attention among queries (post-norm)
    {
        // Q = K = queries + query_pos, V = queries (no pos)
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        const int64_t N = queries->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, q_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, queries), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto* sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);
        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        queries = ggml_add(ctx, queries, sa_out);
        // Post-norm: norm2 (Python's post-SA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm2_w, ly.norm2_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_sa");
        }
    }

    // 2. Cross-attention to text tokens (post-norm)
    {
        // Q = queries + query_pos, K = V = text_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], D * ly.ca_text_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 2 * D * ly.ca_text_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = text_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, text_feats), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, text_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        auto * text_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, N_q, n_heads, B);
        float scale = 1.0f / sqrtf((float)HD);
        auto* ca_out = ggml_flash_attn_ext(ctx, Q, K, V, text_mask, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_text_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_text_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: catext_norm (Python's post-text-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm3_w, ly.norm3_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_text_ca");
        }
    }

    // 3. Cross-attention to conditioned image features (post-norm)
    {
        // Q = queries + query_pos, K = enc_feats + enc_pos, V = enc_feats
        auto* q_in = ggml_add(ctx, queries, query_pos);
        auto* k_in = ggml_add(ctx, enc_feats, enc_pos);

        auto* q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto* k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto* v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
        auto* q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto* k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto* v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q = queries->ne[1];
        const int64_t N_kv = enc_feats->ne[1];
        const int64_t B = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto* K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto* V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, enc_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        struct ggml_tensor * ca_out = nullptr;

        if (rpb_mask) {
            // Keep the box-relative positional bias in fp32. The CPU flash-attn
            // kernel reads mask storage as fp16, which is good enough for token
            // padding masks but introduces avoidable drift here.
            auto * kq = ggml_mul_mat(ctx, K, Q);                    // [N_kv, N_q, NH, B]
            kq = ggml_soft_max_ext(ctx, kq, rpb_mask, scale, 0.0f); // [N_kv, N_q, NH, B]

            auto * v_t = ggml_cont(ctx, ggml_transpose(ctx, V));    // [N_kv, HD, NH, B]
            ca_out = ggml_mul_mat(ctx, v_t, kq);                    // [HD, N_q, NH, B]
            ca_out = ggml_cont(ctx, ggml_permute(ctx, ca_out, 0, 2, 1, 3));
        } else {
            ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        }

        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        queries = ggml_add(ctx, queries, ca_out);
        // Post-norm: norm1 (Python's post-image-CA norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm1_w, ly.norm1_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_after_img_ca");
        }
    }

    // 4. FFN (post-norm, ReLU)
    {
        auto* ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, queries);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        queries = ggml_add(ctx, queries, ffn);
        // Post-norm: norm3 (Python's post-FFN norm)
        queries = sam3_layer_norm(ctx, queries, ly.norm4_w, ly.norm4_b);
        if (layer_idx == 0) {
            ggml_set_name(queries, "ddec_layer0_full_out");
        }
    }

    return queries;
}

// DotProductScoring: classify queries against text features via dot product.
//
// Python reference (DotProductScoring.forward):
//   1. prompt_mlp(prompt) → residual MLP + LN on text features
//   2. mean_pool_text(result, prompt_mask) → pooled [BS, D] (only valid tokens)
//   3. prompt_proj(pooled) → [BS, D]
//   4. hs_proj(hs) → [num_layer, BS, N_q, D]
//   5. matmul(proj_hs, proj_pooled.unsqueeze(-1)) → dot product → [num_layer, BS, N_q, 1]
//   6. scale by 1/sqrt(D)
//   7. clamp to [-12, 12]
//
// query_outputs: [D, N_q, B] — the 200 object query outputs
// text_features: [D, T, B] — text encoder output (already through resizer)
// text_valid_mask: [T, 1, B] — 1.0 for valid tokens, 0.0 for padding (or nullptr for all-valid)
// Returns: class_scores [N_q, B] (one score per query per batch)
static struct ggml_tensor* sam3_dot_product_scoring(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* query_outputs,    // [D, N_q, B]
    struct ggml_tensor* text_features,    // [D, T, B]
    struct ggml_tensor* text_valid_mask)  // [T, 1, B] or nullptr
{
    const auto& tensors = model.tensors;
    const int64_t D = query_outputs->ne[0];  // 256
    const int64_t T = text_features->ne[1];
    const int64_t B = text_features->ne[2];

    // Step 1: Apply prompt_mlp on text features (residual MLP + LayerNorm)
    auto* text_mlp = text_features;  // [D, T, B]
    auto* orig_text = text_features;
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.0.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.0.bias"));
    text_mlp = ggml_relu(ctx, text_mlp);
    text_mlp = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.1.weight"), text_mlp);
    text_mlp = ggml_add(ctx, text_mlp, tensors.at("scoring.prompt_mlp.layers.1.bias"));
    text_mlp = ggml_add(ctx, text_mlp, orig_text);
    text_mlp = sam3_layer_norm(ctx, text_mlp,
                               tensors.at("scoring.prompt_mlp.out_norm.weight"),
                               tensors.at("scoring.prompt_mlp.out_norm.bias"));
    // text_mlp: [D, T, B]
    ggml_set_name(text_mlp, "scoring_prompt_mlp_out");

    // Step 2: Mean-pool over valid text tokens → [D, 1, B]
    // Python: pooled = (prompt * is_valid).sum(0) / num_valid
    struct ggml_tensor* text_pooled;
    if (text_valid_mask) {
        // Permute to [T, D, B] for masked pooling
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));  // [T, D, B]
        // Mask: text_valid_mask is [T, 1, B] — broadcast multiply zeros out padding
        tp = ggml_mul(ctx, tp, text_valid_mask);  // [T, D, B] with padding zeroed
        // Sum over T dimension: pool_1d with SUM kernel=T
        // ggml_pool_1d AVG divides by T; we want SUM then divide by n_valid.
        // Use AVG and then scale by T/n_valid? Or use a manual approach.
        // Simpler: sum via pool_1d with AVG, then scale by T/n_valid.
        // But n_valid is dynamic. Instead: sum = mean * T, then divide by n_valid.
        // We pass n_valid as part of the mask: text_valid_mask sums to n_valid.
        // pool_1d(masked, AVG, T, T, 0) = sum(masked) / T. Multiply by T → sum(masked).
        // Then divide by n_valid. But n_valid is a scalar we know CPU-side.
        // For simplicity: compute AVG over ALL T positions (with padding zeroed out).
        // This gives sum(valid) / T. To get sum(valid) / n_valid, scale by T / n_valid.
        // We embed the scale factor into the mask: mask = (T / n_valid) for valid, 0 for pad.
        // Then AVG(mask * features) = sum(valid * T/n_valid) / T = sum(valid) / n_valid. ✓
        // Caller should set mask values to T/n_valid for valid tokens, 0 for padding.
        auto* pooled_t = ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));  // [D, 1, B]
    } else {
        // All tokens valid — simple mean
        auto* tp = ggml_cont(ctx, ggml_permute(ctx, text_mlp, 1, 0, 2, 3));
        auto* pooled_t = ggml_pool_1d(ctx, tp, GGML_OP_POOL_AVG, (int)T, (int)T, 0);
        text_pooled = ggml_cont(ctx, ggml_permute(ctx, pooled_t, 1, 0, 2, 3));
    }
    ggml_set_name(text_pooled, "scoring_pooled");

    // Step 3: Project pooled prompt through prompt_proj: D→D
    auto* proj_pooled = ggml_mul_mat(ctx, tensors.at("scoring.prompt_proj.weight"), text_pooled);
    proj_pooled = ggml_add(ctx, proj_pooled, tensors.at("scoring.prompt_proj.bias"));
    // proj_pooled: [D, 1, B]
    ggml_set_name(proj_pooled, "scoring_proj_pooled");

    // Step 4: Project queries through hs_proj: D→D
    auto* proj_hs = ggml_mul_mat(ctx, tensors.at("scoring.hs_proj.weight"), query_outputs);
    proj_hs = ggml_add(ctx, proj_hs, tensors.at("scoring.hs_proj.bias"));
    // proj_hs: [D, N_q, B]
    ggml_set_name(proj_hs, "scoring_proj_hs");

    // Step 5: Dot product — for each query, dot with pooled prompt
    // matmul(proj_hs, proj_pooled.unsqueeze(-1)) in Python = batched vector-matrix multiply
    // ggml_mul_mat(A, B) = A^T @ B
    // With A = proj_pooled [D, 1, B], B = proj_hs [D, N_q, B]:
    // result = [1, N_q, B] — each element is dot product of query with pooled prompt
    auto* scores = ggml_mul_mat(ctx, proj_pooled, proj_hs);  // [1, N_q, B]

    // Step 6: Scale by 1/sqrt(D)
    float scale = 1.0f / sqrtf((float)D);
    scores = ggml_scale(ctx, scores, scale);

    // Step 7: Clamp to [-12, 12]
    scores = ggml_clamp(ctx, scores, -12.0f, 12.0f);

    // Reshape to [N_q, B]
    const int64_t N_q = query_outputs->ne[1];
    scores = ggml_reshape_2d(ctx, scores, N_q, B);
    ggml_set_name(scores, "scoring_class_scores");

    return scores;
}

// Build full DETR decoder graph.
// enc_feats: [D, N_kv, B] conditioned features from fusion encoder (N_kv=5184)
// enc_pos: [D, N_kv, B] positional encoding
// text_feats: [D, T, B] text features
// Returns struct with:
//   queries: [D, 201, B] (all query outputs including presence token)
//   pred_boxes: [4, 200, B] (cx, cy, w, h in [0,1])
//   class_scores: [200, B]
//   presence_score: [1, B]
struct sam3_ddec_output {
    struct ggml_tensor* queries;         // [D, 201, B]
    struct ggml_tensor* presence_feats;  // [D, 1, B] pre-decoder-norm presence token
    struct ggml_tensor* pred_boxes;      // [4, 200, B]
    struct ggml_tensor* class_scores;    // [200, B]
    struct ggml_tensor* presence_score;  // [1, B]
};

static sam3_ddec_output sam3_build_ddec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_feats,       // [D, N_kv, B]
    struct ggml_tensor* enc_pos,         // [D, N_kv, B]
    struct ggml_tensor* text_feats,      // [D, T, B]
    struct ggml_tensor* sine_dim_t,      // [1, 64] — pre-computed angle multipliers
    struct ggml_tensor* rpb_coords,      // [feat_hw] — normalized grid coords (or nullptr)
    struct ggml_tensor* text_attn_bias = nullptr,  // [T, 1, B] additive text padding bias
    struct ggml_tensor* text_valid_mask = nullptr)  // [T, 1, B] for scoring (or nullptr)
{
    const auto& hp = model.hparams;
    const auto& tensors = model.tensors;
    const int D = hp.neck_dim;            // 256
    const int NQ = hp.ddec_num_queries;   // 200
    const int B = (int)enc_feats->ne[2];  // batch (1)
    const int feat_hw = hp.n_img_embd();  // 72

    // ── Initialize queries from query_embed ──────────────────────────────
    auto* content = ggml_reshape_3d(ctx, model.ddec.query_embed, D, NQ, 1);
    auto* pres_tok = ggml_reshape_3d(ctx, model.ddec.presence_token, D, 1, 1);
    auto* queries = ggml_concat(ctx, pres_tok, content, 1);  // [D, NQ+1, B=1]

    // Reference points: sigmoid → initial anchor boxes
    auto* ref_pts_raw = ggml_cont(ctx, tensors.at("ddec.reference_points.weight"));  // [4, NQ]
    auto* ref_boxes = ggml_sigmoid(ctx, ref_pts_raw);                // [4, NQ]
    ref_boxes = ggml_reshape_3d(ctx, ref_boxes, 4, NQ, 1);           // [4, NQ, 1]
    auto * ref_boxes_dbg = ggml_cont(ctx, ref_boxes);
    ggml_set_name(ref_boxes_dbg, "ddec_ref_boxes_init");

    // ── Run decoder layers ───────────────────────────────────────────────
    // Per-layer: recompute query_pos from updated ref_boxes (matching Python exactly)
    struct ggml_tensor * last_presence = pres_tok;
    for (int i = 0; i < hp.ddec_layers; ++i) {
        // Recompute query_pos from current ref_boxes via sine embed + ref_point_head MLP
        auto* query_pos = sam3_build_query_pos(ctx, model, ref_boxes, sine_dim_t, i);

        // Compute box-relative positional bias for image cross-attention
        struct ggml_tensor* rpb_mask = nullptr;
        if (rpb_coords) {
            rpb_mask = sam3_compute_box_rpb(ctx, model, ref_boxes, rpb_coords, feat_hw, i);
        }

        queries = sam3_ddec_layer_forward(ctx, model.ddec.layers[i],
                                          queries, query_pos,
                                          enc_feats, enc_pos,
                                          text_feats, hp.ddec_heads,
                                          text_attn_bias,
                                          rpb_mask,
                                          i);

        // Box refinement after each layer (on object queries only, not presence token)
        auto* obj_q = ggml_view_3d(ctx, queries, D, NQ, 1,
                                   queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
        obj_q = ggml_cont(ctx, obj_q);
        sam3_name_tensorf(obj_q, "ddec_layer%d_out", i);

        auto* pres_q = ggml_view_3d(ctx, queries, D, 1, 1,
                                    queries->nb[1], queries->nb[2], 0);
        pres_q = ggml_cont(ctx, pres_q);
        if (i == 0) {
            ggml_set_name(pres_q, "ddec_layer0_presence");
        }
        last_presence = pres_q;

        // Apply the final decoder norm before box refinement (use_normed_output_consistently)
        auto* obj_q_normed = sam3_layer_norm(ctx, obj_q,
                                             tensors.at("ddec.norm.weight"),
                                             tensors.at("ddec.norm.bias"));

        // Shared bbox_embed MLP
        auto* bd = obj_q_normed;
        for (int j = 0; j < 3; ++j) {
            auto wn = "ddec.bbox_embed.layers." + std::to_string(j) + ".weight";
            auto bn = "ddec.bbox_embed.layers." + std::to_string(j) + ".bias";
            bd = ggml_mul_mat(ctx, tensors.at(wn), bd);
            bd = ggml_add(ctx, bd, tensors.at(bn));
            if (j < 2) bd = ggml_relu(ctx, bd);
        }
        // bd: [4, NQ, 1]

        // ref_boxes = sigmoid(inverse_sigmoid(ref_boxes) + box_delta)
        auto* ref_inv_cur = sam3_inverse_sigmoid(ctx, ref_boxes);
        ref_boxes = ggml_sigmoid(ctx, ggml_add(ctx, ref_inv_cur, bd));
        sam3_name_tensorf(ref_boxes, "ddec_layer%d_refboxes", i);
    }

    // ── Final normalization ──────────────────────────────────────────────
    // Match Python: decoder.norm is applied to object queries only.
    auto* obj_queries = ggml_view_3d(ctx, queries, D, NQ, 1,
                                     queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
    obj_queries = ggml_cont(ctx, obj_queries);
    obj_queries = sam3_layer_norm(ctx, obj_queries,
                                  tensors.at("ddec.norm.weight"),
                                  tensors.at("ddec.norm.bias"));
    ggml_set_name(obj_queries, "ddec_normed_output");

    auto * queries_for_seg = ggml_concat(ctx, last_presence, obj_queries, 1);

    auto* class_scores = sam3_dot_product_scoring(ctx, model, obj_queries, text_feats, text_valid_mask);
    // class_scores: [NQ, B]

    // ── Presence score ───────────────────────────────────────────────────
    // Presence token head: LN + 3-layer MLP (D→D→D→1)
    auto* pres_out = sam3_layer_norm(ctx, last_presence,
                               tensors.at("ddec.presence_token_out_norm.weight"),
                               tensors.at("ddec.presence_token_out_norm.bias"));

    for (int j = 0; j < 3; ++j) {
        auto wn = "ddec.presence_token_head.layers." + std::to_string(j) + ".weight";
        auto bn = "ddec.presence_token_head.layers." + std::to_string(j) + ".bias";
        pres_out = ggml_mul_mat(ctx, tensors.at(wn), pres_out);
        pres_out = ggml_add(ctx, pres_out, tensors.at(bn));
        if (j < 2) pres_out = ggml_relu(ctx, pres_out);
    }
    // Keep presence as raw logit (no sigmoid yet — applied during post-processing)
    auto* presence_score = ggml_reshape_2d(ctx, pres_out, 1, 1);
    // presence_score: [1, B] — raw logit

    sam3_ddec_output out;
    out.queries = queries_for_seg;        // [D, NQ+1, B]
    out.presence_feats = last_presence;   // [D, 1, B]
    out.pred_boxes = ref_boxes;           // [4, NQ, B]
    out.class_scores = class_scores;      // [NQ, B]
    out.presence_score = presence_score;  // [1, B]

    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Segmentation head (MaskFormer) — graph building
// ═══════════════════════════════════════════════════════════════════════════════

// Build pixel decoder: progressively upsample FPN features.
// fpn_feats[0]: [D, 288, 288, B] (highest res)
// fpn_feats[1]: [D, 144, 144, B]
// fpn_feats[2]: [D,  72,  72, B] (lowest res)
// Returns: [D, 288, 288, B] pixel features
//
// Python PixelDecoder.forward:
//   prev_fpn = backbone_feats[-1]  (lowest res)
//   for bb_feat in backbone_feats[:-1][::-1]:  (iterate from second-lowest to highest)
//       prev_fpn = bb_feat + F.interpolate(prev_fpn, size=bb_feat.shape[-2:], mode="nearest")
//       prev_fpn = conv_layers[i](prev_fpn)    # conv on the MERGED result
//       prev_fpn = F.relu(norms[i](prev_fpn))  # GroupNorm then ReLU
//
// Python uses GroupNorm(8, 256) — we use ggml_group_norm which normalizes ne[2]
// (the channel dim) in groups.  The conv output is [W, H, D, B] with D in ne[2].
static struct ggml_tensor* sam3_pixel_decoder(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* fpn_feats[3])  // [D, W, H, B] at 3 scales
{
    const auto& seg = model.seg_head;

    // Start from lowest resolution
    auto* feat = fpn_feats[2];  // [D, 72, 72, B]

    // Iteration 0: merge with FPN[1] (144x144)
    // prev_fpn = FPN[1] + upsample(prev_fpn)
    // Permute to [W, H, D, B] for conv operations
    auto* prev = ggml_cont(ctx, ggml_permute(ctx, feat, 2, 0, 1, 3));          // [72, 72, D, B]
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [144, 144, D, B]
    auto* fpn1 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[1], 2, 0, 1, 3));  // [144, 144, D, B]
    prev = ggml_add(ctx, fpn1, prev);                                          // merged
    // Conv 3x3 on the MERGED result (not individual FPN feat)
    prev = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[0], prev);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[0], 1, 1, seg.up_conv_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU — prev is [W, H, D, B] with D in ne[2]
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto * w3d = ggml_reshape_3d(ctx, seg.up_norm_w[0], 1, 1, seg.up_norm_w[0]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto * bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[0], 1, 1, seg.up_norm_b[0]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage0");

    // Iteration 1: merge with FPN[0] (288x288)
    prev = ggml_upscale(ctx, prev, 2, GGML_SCALE_MODE_NEAREST);                // [288, 288, D, B]
    auto* fpn0 = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[0], 2, 0, 1, 3));  // [288, 288, D, B]
    prev = ggml_add(ctx, fpn0, prev);                                          // merged
    // Conv 3x3 on the MERGED result
    prev = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[1], prev);
    {
        auto* b3d = ggml_reshape_3d(ctx, seg.up_conv_b[1], 1, 1, seg.up_conv_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, b3d, prev));
    }
    // GroupNorm(8, 256) then ReLU
    prev = ggml_group_norm(ctx, prev, 8, 1e-5f);
    {
        auto * w3d = ggml_reshape_3d(ctx, seg.up_norm_w[1], 1, 1, seg.up_norm_w[1]->ne[0]);
        prev = ggml_mul(ctx, prev, ggml_repeat(ctx, w3d, prev));
        auto * bn3d = ggml_reshape_3d(ctx, seg.up_norm_b[1], 1, 1, seg.up_norm_b[1]->ne[0]);
        prev = ggml_add(ctx, prev, ggml_repeat(ctx, bn3d, prev));
    }
    prev = ggml_relu(ctx, prev);
    ggml_set_name(prev, "seg_pixel_dec_stage1");

    // Iteration 2 (final): the Python loop has num_upsampling_stages=3, so there are 3 conv layers
    // but only 2 upsample steps (feats[:-1] has 2 elements for 3 total feats).
    // Actually, wait: backbone_feats has 3 entries. backbone_feats[:-1] = first 2.
    // fpn_feats[::-1] iterates [fpn_feats[1], fpn_feats[0]] = 2 iterations, not 3.
    // The 3rd conv layer is NOT used in the PixelDecoder loop — it would need a 4th FPN level.
    // The PixelDecoder creates num_upsampling_stages=3 convs, but only uses 2 in the loop
    // (since there are only 2 upsampling steps for 3 FPN levels).
    // The 3rd conv is unused by the pixel decoder itself.

    // Convert back to [D, W, H, B] layout
    auto * out = ggml_cont(ctx, ggml_permute(ctx, prev, 1, 2, 0, 3));  // [D, 288, 288, B]

    return out;  // [D, 288, 288, B]
}

// Build the full segmentation head graph.
//
// Python UniversalSegmentationHead.forward:
//   1. Cross-attend encoder_hidden_states to prompt → updated encoder
//   2. _embed_pixels: replace lowest-res FPN feat with spatial portion of encoder output
//   3. Run pixel decoder on modified FPN feats
//   4. instance_seg_head (Conv1x1)
//   5. mask_predictor: einsum(mask_embed(queries), instance_embeds)
//
// enc_hidden: [D, N_spatial, B] — fusion encoder output (cross-attended in step 1)
// fpn_feats[3]: the 3 FPN features at different resolutions
// query_outputs: [D, N, B] selected object query outputs
// text_features: [D, T, B] for cross-attention (prompt)
// Returns: mask_logits [W*H, N, B] (raw logits, not sigmoid)
static struct ggml_tensor* sam3_build_seg_head_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* enc_hidden,     // [D, N_spatial, B] fusion encoder output
    struct ggml_tensor* fpn_feats[3],   // FPN features at 3 scales
    struct ggml_tensor* query_outputs,  // [D, N, B]
    struct ggml_tensor* text_features,  // [D, T, B] (for cross-attn, can be nullptr)
    struct ggml_tensor* text_attn_bias = nullptr)
{
    const auto& seg = model.seg_head;
    const auto& tensors = model.tensors;
    const int64_t D = enc_hidden->ne[0];     // 256
    const int64_t B = enc_hidden->ne[2];     // 1
    const int64_t N = query_outputs->ne[1];  // number of selected queries

    // Step 1: Cross-attend encoder hidden states to text/prompt features
    auto* enc = enc_hidden;
    if (text_features) {
        auto* ca_norm = sam3_layer_norm(ctx, enc,
                                        tensors.at("seg.cross_attn_norm.weight"),
                                        tensors.at("seg.cross_attn_norm.bias"));

        auto * ca_mask = sam3_expand_token_attn_bias(ctx, text_attn_bias, enc->ne[1], 8, B);
        auto* ca_out = sam3_multihead_attn_fused(ctx, ca_norm, text_features,
                                                 seg.ca_prompt_q_w, seg.ca_prompt_q_b,
                                                 seg.ca_prompt_out_w, seg.ca_prompt_out_b,
                                                 8, ca_mask);
        enc = ggml_add(ctx, enc, ca_out);
    }
    // enc: [D, N_spatial, B]
    ggml_set_name(enc, "seg_enc_after_ca");

    // Step 2: Replace lowest-res FPN feat with spatial portion of encoder output
    // enc is [D, 5184, B] where 5184 = 72*72. Reshape to [D, 72, 72, B].
    const int64_t feat_hw = model.hparams.n_img_embd();  // 72
    auto* enc_spatial = ggml_reshape_4d(ctx, enc, D, feat_hw, feat_hw, B);
    auto* enc_spatial_dbg = ggml_cont(ctx, ggml_permute(ctx, enc_spatial, 2, 0, 1, 3));
    ggml_set_name(enc_spatial_dbg, "seg_enc_visual");

    // Create modified FPN feats: replace the lowest resolution (index 2) with encoder output
    struct ggml_tensor* modified_fpn[3] = {
        fpn_feats[0],  // [D, 288, 288, B] — unchanged
        fpn_feats[1],  // [D, 144, 144, B] — unchanged
        enc_spatial,   // [D,  72,  72, B] — replaced with encoder output
    };

    // Step 3: Run pixel decoder on modified FPN feats
    auto* pixel_feats = sam3_pixel_decoder(ctx, model, modified_fpn);
    // pixel_feats: [D, 288, 288, B]
    auto * pixel_feats_dbg = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));
    ggml_set_name(pixel_feats_dbg, "seg_pixel_decoder_out");

    const int64_t W = pixel_feats->ne[1];  // 288
    const int64_t H = pixel_feats->ne[2];  // 288

    // Step 4: Instance segmentation head: Conv1x1 on pixel features
    auto* pf_conv = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));  // [W, H, D, B] for conv
    pf_conv = ggml_conv_2d_sk_p0(ctx, tensors.at("seg.instance_seg_head.weight"), pf_conv);
    {
        auto* b3d = ggml_reshape_3d(ctx, tensors.at("seg.instance_seg_head.bias"),
                                    1, 1, tensors.at("seg.instance_seg_head.bias")->ne[0]);
        pf_conv = ggml_add(ctx, pf_conv, ggml_repeat(ctx, b3d, pf_conv));
    }
    auto* pixel_embed = ggml_cont(ctx, ggml_permute(ctx, pf_conv, 1, 2, 0, 3));  // [D, W, H, B]
    auto * pixel_embed_dbg = ggml_cont(ctx, ggml_permute(ctx, pixel_embed, 2, 0, 1, 3));
    ggml_set_name(pixel_embed_dbg, "seg_instance_embed");

    // Step 5: Mask embedding: project query outputs through mask_embed MLP
    // 3-layer MLP: D→D→D→D (each layer registered separately)
    auto* mask_embed = query_outputs;
    for (int j = 0; j < 3; ++j) {
        auto wn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".weight";
        auto bn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".bias";
        mask_embed = ggml_mul_mat(ctx, tensors.at(wn), mask_embed);
        mask_embed = ggml_add(ctx, mask_embed, tensors.at(bn));
        if (j < 2) mask_embed = ggml_relu(ctx, mask_embed);
    }
    // mask_embed: [D, N, B]
    ggml_set_name(mask_embed, "seg_mask_embed");

    // Mask prediction: einsum('bqc,bchw->bqhw') = mask_embed^T @ pixel_embed
    // Flatten pixel_embed: [D, W*H, B]
    auto* pe_flat = ggml_reshape_3d(ctx, pixel_embed, D, W * H, B);
    // ggml_mul_mat(A, B) = A^T @ B.  With A=[D, W*H, B], B=[D, N, B]:
    //   A^T is [W*H, D], result = [W*H, N, B].
    auto* masks = ggml_mul_mat(ctx, pe_flat, mask_embed);
    // masks: [W*H, N, B]
    ggml_set_name(masks, "seg_mask_logits");

    return masks;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Memory encoder (Phase 7, Step 7.1)
// ═══════════════════════════════════════════════════════════════════════════════

// CXBlock: depthwise conv + LayerNorm + pointwise MLP with residual scaling.
static struct ggml_tensor* sam3_cxblock_forward(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,        // [D, H, W, B]
    struct ggml_tensor*  dw_w,     // [7, 7, 1, D] depthwise
    struct ggml_tensor*  dw_b,     // [D]
    struct ggml_tensor*  norm_w,   // [D]
    struct ggml_tensor*  norm_b,   // [D]
    struct ggml_tensor*  fc1_w,    // [D, 1024]
    struct ggml_tensor*  fc1_b,    // [1024]
    struct ggml_tensor*  fc2_w,    // [1024, D]
    struct ggml_tensor*  fc2_b,    // [D]
    struct ggml_tensor*  gamma)    // [D]
{
    const int D = (int)x->ne[0];
    const int H = (int)x->ne[1];
    const int W = (int)x->ne[2];

    // Depthwise conv (groups = D): pad=3 for 7x7 kernel
    auto* h = ggml_conv_2d(ctx, dw_w, x, 1, 1, 3, 3, D, 1);
    h = ggml_add(ctx, h, ggml_reshape_4d(ctx, dw_b, 1, 1, D, 1));

    // LayerNorm2d
    h = sam3_layer_norm_2d(ctx, h, norm_w, norm_b);

    // Pointwise MLP: reshape to [D, H*W, B], apply FC, reshape back
    auto* flat = ggml_reshape_3d(ctx, h, D, H * W, 1);
    flat = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, flat), fc1_b);
    flat = ggml_gelu(ctx, flat);
    flat = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, flat), fc2_b);
    h = ggml_reshape_4d(ctx, flat, D, H, W, 1);

    // Residual with learnable scaling: x + gamma * h
    auto* gamma_4d = ggml_reshape_4d(ctx, gamma, D, 1, 1, 1);
    h = ggml_mul(ctx, h, gamma_4d);
    return ggml_add(ctx, x, h);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Memory attention (Phase 7, Step 7.2)
// ═══════════════════════════════════════════════════════════════════════════════

// Build memory attention graph.
// curr_tokens: [D, N, 1] — current frame image tokens (flattened from 72x72 = 5184)
// mem_feats: [MD, M, 1] — concatenated memory features from all memory slots
// obj_ptrs: [D, P, 1] — object pointer tokens (appended to KV) or nullptr
// Returns: conditioned tokens [D, N, 1]
static struct ggml_tensor* sam3_build_mem_attn_graph(
    struct ggml_context* ctx,
    const sam3_model&    model,
    struct ggml_tensor*  curr_tokens,   // [D, N, 1]
    struct ggml_tensor*  mem_feats,     // [MD, M, 1]
    struct ggml_tensor*  obj_ptrs)      // [D, P, 1] or nullptr
{
    const auto& ma = model.mem_attn;
    const int D = model.hparams.neck_dim;  // 256
    const int N = (int)curr_tokens->ne[1];

    auto* x = curr_tokens;  // [D, N, 1]

    for (int l = 0; l < (int)ma.layers.size(); ++l) {
        const auto& ly = ma.layers[l];

        // ── Self-attention with RoPE ──────────────────────────────────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
            auto* q = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_q_w, x_norm), ly.sa_q_b);
            auto* k = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_k_w, x_norm), ly.sa_k_b);
            auto* v = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_v_w, x_norm), ly.sa_v_b);

            // Single-head attention (256-dim)
            q = ggml_reshape_4d(ctx, q, D, 1, N, 1);
            k = ggml_reshape_4d(ctx, k, D, 1, N, 1);
            v = ggml_reshape_4d(ctx, v, D, 1, N, 1);
            q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

            float scale = 1.0f / sqrtf((float)D);
            auto* sa_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            sa_out = ggml_reshape_3d(ctx, sa_out, D, N, 1);
            sa_out = ggml_add(ctx, ggml_mul_mat(ctx, ly.sa_out_w, sa_out), ly.sa_out_b);
            x = ggml_add(ctx, x, sa_out);
        }

        // ── Cross-attention to memory (kv_dim=64) ────────────────────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
            auto* q = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_q_w, x_norm), ly.ca_q_b);

            // KV from memory: project from 64-dim to 256-dim
            auto* k = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_k_w, mem_feats), ly.ca_k_b);
            auto* v = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_v_w, mem_feats), ly.ca_v_b);

            // Concatenate object pointers directly to KV (already D-dim)
            if (obj_ptrs) {
                auto* obj_ptrs_2d = ggml_reshape_2d(ctx, obj_ptrs, D, (int)obj_ptrs->ne[1]);
                k = ggml_concat(ctx, k, obj_ptrs_2d, 1);
                v = ggml_concat(ctx, v, obj_ptrs_2d, 1);
            }

            int M_total = (int)k->ne[1];

            q = ggml_reshape_4d(ctx, q, D, 1, N, 1);
            k = ggml_reshape_4d(ctx, k, D, 1, M_total, 1);
            v = ggml_reshape_4d(ctx, v, D, 1, M_total, 1);
            q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

            float scale = 1.0f / sqrtf((float)D);
            auto* ca_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            ca_out = ggml_reshape_3d(ctx, ca_out, D, N, 1);
            ca_out = ggml_add(ctx, ggml_mul_mat(ctx, ly.ca_out_w, ca_out), ly.ca_out_b);
            x = ggml_add(ctx, x, ca_out);
        }

        // ── FFN ───────────────────────────────────────────────────────────
        {
            auto* x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);
            auto* ffn = ggml_add(ctx, ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm), ly.ffn_fc1_b);
            ffn = ggml_relu(ctx, ffn);
            ffn = ggml_add(ctx, ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn), ly.ffn_fc2_b);
            x = ggml_add(ctx, x, ffn);
        }
    }

    // Final norm
    auto* norm_w = model.tensors.at("mem_attn.norm.weight");
    auto* norm_b = model.tensors.at("mem_attn.norm.bias");
    x = sam3_layer_norm(ctx, x, norm_w, norm_b);

    return x;  // [D, N, 1]
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Object pointer extraction (Phase 7, Step 7.3)
// ═══════════════════════════════════════════════════════════════════════════════

// Extract object pointer from SAM output token via 3-layer MLP (CPU-side).
static void sam3_extract_obj_ptr_cpu(
    const sam3_model& model,
    const float*      sam_token_data,  // [D]
    float             obj_score,
    float*            out_ptr)         // [D]
{
    const int D = model.hparams.neck_dim;
    const float occlusion_threshold = 0.0f;

    if (obj_score < occlusion_threshold) {
        ggml_backend_tensor_get(model.no_obj_ptr, out_ptr, 0, D * sizeof(float));
        return;
    }

    std::vector<float> h(D), tmp(D);
    std::copy(sam_token_data, sam_token_data + D, h.data());

    for (int j = 0; j < 3; ++j) {
        auto* w = model.obj_ptr_proj_w[j];
        auto* b = model.obj_ptr_proj_b[j];

        int nel_w = (int)(w->ne[0] * w->ne[1]);
        std::vector<float> w_data(nel_w);
        if (w->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> w16(nel_w);
            ggml_backend_tensor_get(w, w16.data(), 0, nel_w * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(w16.data(), w_data.data(), nel_w);
        } else {
            ggml_backend_tensor_get(w, w_data.data(), 0, nel_w * sizeof(float));
        }

        std::vector<float> b_data(D);
        ggml_backend_tensor_get(b, b_data.data(), 0, D * sizeof(float));

        for (int o = 0; o < D; ++o) {
            float sum = b_data[o];
            for (int i = 0; i < D; ++i) {
                sum += w_data[o * D + i] * h[i];
            }
            tmp[o] = (j < 2) ? std::max(0.0f, sum) : sum;
        }
        std::swap(h, tmp);
    }
    std::copy(h.begin(), h.end(), out_ptr);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tracker infrastructure (Phase 7, Step 7.4)
// ═══════════════════════════════════════════════════════════════════════════════

// Select memory frames for propagation (most recent + evenly spaced).
static std::vector<int> sam3_select_memory_frames(
    const std::vector<sam3_memory_slot>& bank,
    int                                   max_slots)
{
    if ((int)bank.size() <= max_slots) {
        std::vector<int> all(bank.size());
        for (int i = 0; i < (int)bank.size(); ++i) all[i] = i;
        return all;
    }
    std::vector<int> selected;
    selected.push_back(0);
    selected.push_back((int)bank.size() - 1);
    int remaining = max_slots - 2;
    if (remaining > 0) {
        float step = (float)(bank.size() - 2) / (remaining + 1);
        for (int i = 0; i < remaining; ++i) {
            int idx = 1 + (int)((i + 1) * step);
            idx = std::min(idx, (int)bank.size() - 2);
            selected.push_back(idx);
        }
    }
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

// Compute mask IoU between two binary masks.
static float sam3_mask_iou(const uint8_t* a, const uint8_t* b, int n) {
    int inter = 0, uni = 0;
    for (int i = 0; i < n; ++i) {
        bool va = a[i] > 127;
        bool vb = b[i] > 127;
        if (va && vb) ++inter;
        if (va || vb) ++uni;
    }
    return (uni > 0) ? (float)inter / uni : 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Post-processing: hole filling and sprinkle removal (Phase 7, Step 7.8)
// ═══════════════════════════════════════════════════════════════════════════════

// Fill small holes in a binary mask using BFS connected components.
static void sam3_fill_holes(uint8_t* mask, int w, int h, int area_threshold) {
    const int n = w * h;
    std::vector<int> labels(n, -1);
    int next_label = 0;
    std::vector<int> component_sizes;

    for (int i = 0; i < n; ++i) {
        if (mask[i] > 127 || labels[i] >= 0) continue;
        int label = next_label++;
        component_sizes.push_back(0);
        std::vector<int> queue;
        queue.push_back(i);
        labels[i] = label;
        int head = 0;
        bool touches_border = false;
        while (head < (int)queue.size()) {
            int p = queue[head++];
            component_sizes[label]++;
            int px = p % w, py = p / w;
            if (px == 0 || px == w - 1 || py == 0 || py == h - 1) touches_border = true;
            int dx[] = {-1, 1, 0, 0};
            int dy[] = {0, 0, -1, 1};
            for (int d = 0; d < 4; ++d) {
                int nx2 = px + dx[d], ny2 = py + dy[d];
                if (nx2 < 0 || nx2 >= w || ny2 < 0 || ny2 >= h) continue;
                int ni = ny2 * w + nx2;
                if (mask[ni] <= 127 && labels[ni] < 0) {
                    labels[ni] = label;
                    queue.push_back(ni);
                }
            }
        }
        if (touches_border) component_sizes[label] = area_threshold + 1;
    }
    for (int i = 0; i < n; ++i) {
        if (mask[i] <= 127 && labels[i] >= 0 && component_sizes[labels[i]] <= area_threshold) {
            mask[i] = 255;
        }
    }
}

// Remove small foreground sprinkles.
static void sam3_remove_sprinkles(uint8_t* mask, int w, int h, int area_threshold) {
    const int n = w * h;
    std::vector<int> labels(n, -1);
    int next_label = 0;
    std::vector<int> component_sizes;

    for (int i = 0; i < n; ++i) {
        if (mask[i] <= 127 || labels[i] >= 0) continue;
        int label = next_label++;
        component_sizes.push_back(0);
        std::vector<int> queue;
        queue.push_back(i);
        labels[i] = label;
        int head = 0;
        while (head < (int)queue.size()) {
            int p = queue[head++];
            component_sizes[label]++;
            int px = p % w, py = p / w;
            int dx[] = {-1, 1, 0, 0};
            int dy[] = {0, 0, -1, 1};
            for (int d = 0; d < 4; ++d) {
                int nx2 = px + dx[d], ny2 = py + dy[d];
                if (nx2 < 0 || nx2 >= w || ny2 < 0 || ny2 >= h) continue;
                int ni = ny2 * w + nx2;
                if (mask[ni] > 127 && labels[ni] < 0) {
                    labels[ni] = label;
                    queue.push_back(ni);
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (mask[i] > 127 && labels[i] >= 0 && component_sizes[labels[i]] <= area_threshold) {
            mask[i] = 0;
        }
    }
}

// Resolve overlapping masks: higher-scoring instances take priority.
static void sam3_resolve_overlaps(std::vector<sam3_detection>& dets) {
    if (dets.size() <= 1) return;
    const int w = dets[0].mask.width;
    const int h = dets[0].mask.height;
    if (w == 0 || h == 0) return;
    std::sort(dets.begin(), dets.end(), [](const sam3_detection& a, const sam3_detection& b) {
        return a.score > b.score;
    });
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        bool claimed = false;
        for (auto& det : dets) {
            if (det.mask.data.empty()) continue;
            if (claimed) {
                det.mask.data[i] = 0;
            } else if (det.mask.data[i] > 127) {
                claimed = true;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Post-processing: NMS, bilinear interpolation, mask binarization
// ═══════════════════════════════════════════════════════════════════════════════

// Compute IoU between two boxes [x0, y0, x1, y1].
static float sam3_box_iou(const sam3_box& a, const sam3_box& b) {
    float x0 = std::max(a.x0, b.x0);
    float y0 = std::max(a.y0, b.y0);
    float x1 = std::min(a.x1, b.x1);
    float y1 = std::min(a.y1, b.y1);
    float inter = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
    float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);
    float uni = area_a + area_b - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

// Non-maximum suppression on detections, sorted by score descending.
// Returns indices of kept detections.
static std::vector<int> sam3_nms(const std::vector<sam3_detection>& dets, float iou_thresh) {
    // Sort indices by score descending
    std::vector<int> indices(dets.size());
    for (int i = 0; i < (int)dets.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int> keep;

    for (int idx : indices) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (int j : indices) {
            if (suppressed[j] || j == idx) continue;
            if (sam3_box_iou(dets[idx].box, dets[j].box) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

// Bilinear interpolation of a flat mask [H_in * W_in] to [H_out * W_out].
static std::vector<float> sam3_bilinear_interpolate(const float* src, int src_w, int src_h,
                                                    int dst_w, int dst_h) {
    std::vector<float> dst(dst_w * dst_h);
    const float sx = (float)src_w / dst_w;
    const float sy = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int y0 = std::max(0, (int)fy);
        const int y1 = std::min(src_h - 1, y0 + 1);
        const float wy = fy - y0;

        for (int x = 0; x < dst_w; ++x) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int x0 = std::max(0, (int)fx);
            const int x1 = std::min(src_w - 1, x0 + 1);
            const float wx = fx - x0;

            float v = (1 - wy) * ((1 - wx) * src[y0 * src_w + x0] + wx * src[y0 * src_w + x1]) + wy * ((1 - wx) * src[y1 * src_w + x0] + wx * src[y1 * src_w + x1]);
            dst[y * dst_w + x] = v;
        }
    }
    return dst;
}

// Convert (cx, cy, w, h) in [0,1] to (x0, y0, x1, y1) in pixel coordinates.
static sam3_box sam3_cxcywh_to_xyxy(float cx, float cy, float w, float h,
                                    int img_w, int img_h) {
    sam3_box box;
    box.x0 = (cx - w * 0.5f) * img_w;
    box.y0 = (cy - h * 0.5f) * img_h;
    box.x1 = (cx + w * 0.5f) * img_w;
    box.y1 = (cy + h * 0.5f) * img_h;
    // Clamp to image bounds
    box.x0 = std::max(0.0f, std::min(box.x0, (float)img_w));
    box.y0 = std::max(0.0f, std::min(box.y0, (float)img_h));
    box.x1 = std::max(0.0f, std::min(box.x1, (float)img_w));
    box.y1 = std::max(0.0f, std::min(box.y1, (float)img_h));
    return box;
}

// Compute sinusoidal positional embedding for DETR reference points.
// ref_boxes: [NQ, 4] (cx, cy, w, h) — returns [NQ, 512] embedding.
//
// Python gen_sineembed_for_position output order for 4D input:
//   [pos_y, pos_x, pos_w, pos_h]  (y FIRST, then x, then w, h)
// where pos_tensor[:, :, 0] = x = cx, pos_tensor[:, :, 1] = y = cy
// So the 512-dim vector is: [cy_embed_128, cx_embed_128, w_embed_128, h_embed_128]
static void sam3_sine_pos_embed_boxes(float* out, const float* boxes, int NQ, int num_feats) {
    const float temperature = 10000.0f;
    const int feats_per_coord = num_feats;  // 128 features per coordinate

    // Map coordinate indices to match Python output order: [y, x, w, h] = [cy, cx, w, h]
    // boxes layout: [cx(0), cy(1), w(2), h(3)]
    // Python output: slot 0 = cy (idx 1), slot 1 = cx (idx 0), slot 2 = w (idx 2), slot 3 = h (idx 3)
    const int coord_order[4] = {1, 0, 2, 3};  // map output slot → boxes index

    for (int q = 0; q < NQ; ++q) {
        for (int slot = 0; slot < 4; ++slot) {
            float val = boxes[q * 4 + coord_order[slot]];
            for (int i = 0; i < feats_per_coord; ++i) {
                int paired = i & ~1;
                float dim_t = powf(temperature, (float)paired / (float)feats_per_coord);
                float angle = val * 2.0f * (float)M_PI / dim_t;
                if (i % 2 == 0) {
                    out[q * feats_per_coord * 4 + slot * feats_per_coord + i] = sinf(angle);
                } else {
                    out[q * feats_per_coord * 4 + slot * feats_per_coord + i] = cosf(angle);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image segmentation — PCS (text-prompted)
// ═══════════════════════════════════════════════════════════════════════════════

sam3_result sam3_segment_pcs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pcs_params& params) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto& hp = model.hparams;
    const int D = hp.neck_dim;           // 256
    const int H = hp.n_img_embd();       // 72
    const int L = hp.text_ctx_len;       // 32
    const int NQ = hp.ddec_num_queries;  // 200
    sam3_result result;

    // ── Check that image has been encoded ────────────────────────────────
    if (!state.neck_det[0]) {
        fprintf(stderr, "%s: image not encoded — call sam3_encode_image first\n", __func__);
        return result;
    }

    // ── Tokenize text prompt ─────────────────────────────────────────────
    auto token_ids = sam3_tokenize(const_cast<sam3_bpe_tokenizer&>(model.tokenizer),
                                   params.text_prompt, L);
    if (token_ids.empty()) {
        fprintf(stderr, "%s: failed to tokenize text prompt\n", __func__);
        return result;
    }

    fprintf(stderr, "%s: text='%s', %zu tokens\n", __func__,
            params.text_prompt.c_str(), token_ids.size());

    // ── Build computation graph ──────────────────────────────────────────
    const size_t buf_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return result;
    }

    // ── Text encoder input ───────────────────────────────────────────────
    auto* inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    // ── Text encoder graph (implemented in Phase 4) ─────────────────────
    // sam3_build_text_encoder_graph creates causal mask internally (named "causal_mask").
    // Returns: [256, L] = [D, 32] (2D tensor).
    auto* text_features_2d = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    // Reshape to [D, L, 1] for consistent 3D processing in fusion/DETR
    auto* text_features = ggml_reshape_3d(ctx0, text_features_2d, D, L, 1);
    ggml_set_name(text_features, "text_features");

    // ── Prepare image features for fusion encoder ────────────────────────
    // Use the 72×72 neck features (scale 2) for the fusion encoder
    // neck_det[2]: [D, 72, 72, B] — flatten spatial dims → [D, 5184, 1]
    auto* img_feats = ggml_reshape_3d(ctx0, state.neck_det[2], D, H * H, 1);
    // PE for image features: flatten from [D, 72, 72, 1] → [D, 5184, 1]
    auto* img_pe = ggml_reshape_3d(ctx0, state.neck_det_pe[2], D, H * H, 1);

    // ── Fusion encoder graph ─────────────────────────────────────────────
    // ── DETR decoder inputs: sine dim_t for positional encoding, RPB grid ─
    // sine_dim_t: [1, 64] — pre-computed 2π / 10000^(2i/128) for i=0..63
    auto* sine_dim_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 64);
    ggml_set_name(sine_dim_t, "sine_dim_t");
    ggml_set_input(sine_dim_t);

    // RPB coordinate grid: [feat_hw] = [72] — normalized [0/72, 1/72, ..., 71/72]
    auto* rpb_coords = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, H);
    ggml_set_name(rpb_coords, "rpb_coords");
    ggml_set_input(rpb_coords);

    // Text validity mask for DotProductScoring: [L, 1, 1]
    // Values: T/n_valid for valid tokens, 0 for padding
    auto* text_valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_valid_mask, "text_valid_mask");
    ggml_set_input(text_valid_mask);

    // Text attention bias for prompt cross-attention: 0 for valid tokens, a
    // large negative bias for padding tokens.
    auto* text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_attn_bias, "text_attn_bias");
    ggml_set_input(text_attn_bias);

    // ── Fusion encoder graph ─────────────────────────────────────────────
    auto* conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features,
                                              img_pe, text_attn_bias);
    ggml_set_name(conditioned, "fenc_output");
    // conditioned: [D, 5184, 1]

    // ── DETR decoder graph ───────────────────────────────────────────────
    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features,
                                          sine_dim_t, rpb_coords, text_attn_bias, text_valid_mask);
    ggml_set_name(ddec_out.class_scores, "class_scores");
    ggml_set_name(ddec_out.pred_boxes, "pred_boxes");
    ggml_set_name(ddec_out.presence_score, "presence_score");
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    // ── Segmentation head graph ──────────────────────────────────────────
    // Use FPN features from detector neck at 3 scales
    struct ggml_tensor* fpn_feats[3] = {
        state.neck_det[0],  // [D, 288, 288, B]
        state.neck_det[1],  // [D, 144, 144, B]
        state.neck_det[2],  // [D,  72,  72, B]
    };

    // Extract object queries from DETR output (skip presence token)
    auto* obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                     ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                     1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    // Pass fusion encoder output (conditioned) for pixel decoder integration
    // The seg head will cross-attend enc output to text, then replace lowest-res FPN feat,
    // then run the pixel decoder, then produce masks.
    auto* mask_logits = sam3_build_seg_head_graph(ctx0, model, conditioned, fpn_feats,
                                                  obj_queries, text_features, text_attn_bias);
    ggml_set_name(mask_logits, "mask_logits");
    ggml_set_output(mask_logits);
    // mask_logits: [288*288, NQ, 1]  (per-query masks are contiguous)

    // ── Build and allocate graph ─────────────────────────────────────────
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }

    fprintf(stderr, "%s: graph allocated, %d nodes\n", __func__, ggml_graph_n_nodes(graph));

    // ── Set input data ───────────────────────────────────────────────────
    // Token IDs
    ggml_backend_tensor_set(inp_tokens, token_ids.data(), 0, L * sizeof(int32_t));

    // Causal mask (created internally by sam3_build_text_encoder_graph, F16 format)
    {
        auto* causal_mask = ggml_get_tensor(ctx0, "causal_mask");
        if (causal_mask) {
            std::vector<ggml_fp16_t> mask_data(L * L);
            sam3_fill_causal_mask(mask_data.data(), L);
            ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));
        }
    }

    // Sine dim_t for DETR positional encoding: 2π / 10000^(2i/128) for i=0..63
    {
        auto* sdt = ggml_get_tensor(ctx0, "sine_dim_t");
        if (sdt) {
            std::vector<float> dim_t_data(64);
            for (int i = 0; i < 64; ++i) {
                dim_t_data[i] = 2.0f * (float)M_PI / powf(10000.0f, 2.0f * (float)i / 128.0f);
            }
            ggml_backend_tensor_set(sdt, dim_t_data.data(), 0, 64 * sizeof(float));
        }
    }

    // RPB coordinate grid: [0/72, 1/72, ..., 71/72]
    {
        auto* rc = ggml_get_tensor(ctx0, "rpb_coords");
        if (rc) {
            std::vector<float> coords(H);
            for (int i = 0; i < H; ++i) coords[i] = (float)i / (float)H;
            ggml_backend_tensor_set(rc, coords.data(), 0, H * sizeof(float));
        }
    }

    // Presence token positional encoding: zeros
    {
        auto* qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
        if (qpos_pres) {
            std::vector<float> zeros(D, 0.0f);
            ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
        }
    }

    // RPB presence token mask: zeros
    {
        auto* rpb_pz = ggml_get_tensor(ctx0, "rpb_pres_zeros");
        if (rpb_pz) {
            int n = (int)(rpb_pz->ne[0] * rpb_pz->ne[1] * rpb_pz->ne[2] * rpb_pz->ne[3]);
            std::vector<float> zeros(n, 0.0f);
            ggml_backend_tensor_set(rpb_pz, zeros.data(), 0, n * sizeof(float));
        }
    }

    // Text validity mask for DotProductScoring
    // Tokens: [SOT, bpe..., EOT, 0, 0, ...]. Valid = non-zero. Pad = 0.
    {
        auto* tvm = ggml_get_tensor(ctx0, "text_valid_mask");
        if (tvm) {
            int n_valid = 0;
            for (int i = 0; i < L; ++i) {
                if (token_ids[i] != 0) ++n_valid;
            }
            if (n_valid == 0) n_valid = 1;  // safety
            float scale = (float)L / (float)n_valid;  // compensate AVG pooling
            std::vector<float> mask_data(L);
            for (int i = 0; i < L; ++i) {
                mask_data[i] = (token_ids[i] != 0) ? scale : 0.0f;
            }
            ggml_backend_tensor_set(tvm, mask_data.data(), 0, L * sizeof(float));
        }
    }

    // Text attention bias for prompt cross-attention.
    {
        auto * tab = ggml_get_tensor(ctx0, "text_attn_bias");
        if (tab) {
            std::vector<float> bias_data(L);
            for (int i = 0; i < L; ++i) {
                bias_data[i] = (token_ids[i] != 0) ? 0.0f : -1.0e9f;
            }
            ggml_backend_tensor_set(tab, bias_data.data(), 0, L * sizeof(float));
        }
    }

    // ── Compute ──────────────────────────────────────────────────────────
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        sam3_graph_compute(model.backend, graph, state.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "%s: graph computed in %.1f ms (%d threads)\n",
                __func__, ms, state.n_threads);
    }

    // ── Read outputs ─────────────────────────────────────────────────────
    std::vector<float> scores_data(NQ);
    ggml_backend_tensor_get(ddec_out.class_scores, scores_data.data(), 0, NQ * sizeof(float));

    std::vector<float> boxes_data(4 * NQ);
    ggml_backend_tensor_get(ddec_out.pred_boxes, boxes_data.data(), 0, 4 * NQ * sizeof(float));

    std::vector<float> pres_data(1);
    ggml_backend_tensor_get(ddec_out.presence_score, pres_data.data(), 0, sizeof(float));

    // presence_score is a raw logit — apply sigmoid to get probability
    float presence_logit = pres_data[0];
    float presence_prob = 1.0f / (1.0f + expf(-presence_logit));

    // Read mask logits: [288*288, NQ, 1] — per-query masks are contiguous
    const int mask_hw = 288;
    std::vector<float> all_masks(NQ * mask_hw * mask_hw);
    ggml_backend_tensor_get(mask_logits, all_masks.data(), 0, all_masks.size() * sizeof(float));

    // ── Post-process: score thresholding ─────────────────────────────────
    // Python joint scoring: outputs_class = inverse_sigmoid(class_scores.sigmoid() * presence.sigmoid())
    // Then final score = outputs_class.sigmoid() = class_scores.sigmoid() * presence.sigmoid()
    // So effectively: score = sigmoid(class_logit) * sigmoid(presence_logit)
    std::vector<sam3_detection> dets;
    for (int q = 0; q < NQ; ++q) {
        float class_prob = 1.0f / (1.0f + expf(-scores_data[q]));
        float score = class_prob * presence_prob;
        if (score < params.score_threshold) continue;

        sam3_detection det;
        // boxes are [4, NQ] in ggml layout: (cx, cy, w, h) per query
        // boxes_data is flat [4*NQ], ggml stores [ne[0]=4, ne[1]=NQ] column-major
        // So box for query q: boxes_data[0 + q*4] through boxes_data[3 + q*4]
        // Wait — ggml reads with ggml_backend_tensor_get as flat bytes.
        // Tensor is [4, NQ, 1] with ne[0]=4. So element (i, q) = data[i + q*4].
        float cx = boxes_data[0 + q * 4];
        float cy = boxes_data[1 + q * 4];
        float bw = boxes_data[2 + q * 4];
        float bh = boxes_data[3 + q * 4];

        det.box = sam3_cxcywh_to_xyxy(cx, cy, bw, bh, state.orig_width, state.orig_height);
        det.score = score;

        // Extract mask for this query and resize to original image size
        const float* mask_ptr = all_masks.data() + q * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);

        // Binarize mask at threshold 0.0 (sigmoid > 0.5 ↔ logit > 0.0)
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i) {
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        }
        det.mask.iou_score = score;

        dets.push_back(std::move(det));
    }

    fprintf(stderr, "%s: %zu detections above threshold %.2f (presence=%.3f, logit=%.3f)\n",
            __func__, dets.size(), params.score_threshold, presence_prob, presence_logit);

    // ── NMS ──────────────────────────────────────────────────────────────
    auto keep = sam3_nms(dets, params.nms_threshold);
    for (int i = 0; i < (int)keep.size(); ++i) {
        dets[keep[i]].instance_id = i + 1;
        result.detections.push_back(std::move(dets[keep[i]]));
    }

    fprintf(stderr, "%s: %zu detections after NMS\n", __func__, result.detections.size());

    // ── Cleanup ──────────────────────────────────────────────────────────
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stderr, "%s: completed in %.1f ms\n", __func__, total_ms);

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SAM attention helper (separate Q, K, V weight/bias)
// ═══════════════════════════════════════════════════════════════════════════════

static struct ggml_tensor* sam3_sam_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q_in,  // [D, N_q, B]
    struct ggml_tensor* k_in,  // [D, N_kv, B]
    struct ggml_tensor* v_in,  // [D, N_kv, B]
    const sam3_sam_attn& attn,
    int n_heads) {
    const int64_t N_q = q_in->ne[1];
    const int64_t B = q_in->ne[2];
    const int64_t N_kv = k_in->ne[1];

    // Project
    auto* Q = ggml_add(ctx, ggml_mul_mat(ctx, attn.q_w, q_in), attn.q_b);
    auto* K = ggml_add(ctx, ggml_mul_mat(ctx, attn.k_w, k_in), attn.k_b);
    auto* V = ggml_add(ctx, ggml_mul_mat(ctx, attn.v_w, v_in), attn.v_b);

    // internal_dim = out_proj cols = attn.q_w->ne[1]
    const int64_t ID = attn.q_w->ne[1];
    const int64_t HD = ID / n_heads;

    // Reshape to multi-head: [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    // Attention
    float scale = 1.0f / sqrtf((float)HD);
    auto* out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);

    // Merge heads: [ID, N_q, B]
    out = ggml_reshape_3d(ctx, out, ID, N_q, B);

    // Output projection
    out = ggml_mul_mat(ctx, attn.out_w, out);
    out = ggml_add(ctx, out, attn.out_b);

    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SAM prompt encoder — graph building (Phase 6, Step 6.1)
// ═══════════════════════════════════════════════════════════════════════════════

// Random Fourier positional encoding for a single (x, y) coordinate
// coords_norm: normalized to [0, 1], pe_gaussian: PyTorch row-major [2, 128]
// Output: [256] = [sin(128); cos(128)]
static void sam3_pe_encode_coord(float* out, float x_norm, float y_norm,
                                 const float* pe_gauss, int num_pos_feats) {
    // Map [0,1] → [-1,1]
    float coords[2] = {2.0f * x_norm - 1.0f, 2.0f * y_norm - 1.0f};

    // coords @ pe_gaussian → [128]
    for (int i = 0; i < num_pos_feats; ++i) {
        float dot = coords[0] * pe_gauss[i] +
                    coords[1] * pe_gauss[num_pos_feats + i];
        dot *= 2.0f * (float)M_PI;
        out[i] = sinf(dot);
        out[i + num_pos_feats] = cosf(dot);
    }
}

// Read SAM prompt encoder weights from GPU and cache them in state.
// Also pre-computes the dense PE grid and no-mask tiled embedding.
// These never change between PVS calls for the same model.
static void sam3_populate_pe_cache(sam3_state & state, const sam3_model & model) {
    if (state.pe_cache_valid) return;

    const int D = model.hparams.sam_embed_dim;      // 256
    const int H = model.hparams.n_img_embd();        // 72
    const int num_pos_feats = D / 2;                  // 128
    const int pe_nel = 2 * num_pos_feats;             // 256
    const auto & pe = model.sam_pe;

    // pe_gaussian
    state.pe_gauss_cache.resize(pe_nel);
    if (pe.pe_gaussian->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(pe_nel);
        ggml_backend_tensor_get(pe.pe_gaussian, tmp.data(), 0, pe_nel * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.pe_gauss_cache.data(), pe_nel);
    } else {
        ggml_backend_tensor_get(pe.pe_gaussian, state.pe_gauss_cache.data(), 0, pe_nel * sizeof(float));
    }

    // point_embed[4]
    for (int i = 0; i < 4; ++i) {
        if (pe.point_embed[i]->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(D);
            ggml_backend_tensor_get(pe.point_embed[i], tmp.data(), 0, D * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), state.point_emb_cache[i], D);
        } else {
            ggml_backend_tensor_get(pe.point_embed[i], state.point_emb_cache[i], 0, D * sizeof(float));
        }
    }

    // not_a_point_embed
    if (pe.not_a_point_embed->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(D);
        ggml_backend_tensor_get(pe.not_a_point_embed, tmp.data(), 0, D * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.not_a_point_cache, D);
    } else {
        ggml_backend_tensor_get(pe.not_a_point_embed, state.not_a_point_cache, 0, D * sizeof(float));
    }

    // no_mask_embed
    if (pe.no_mask_embed->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(D);
        ggml_backend_tensor_get(pe.no_mask_embed, tmp.data(), 0, D * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), state.no_mask_emb_cache, D);
    } else {
        ggml_backend_tensor_get(pe.no_mask_embed, state.no_mask_emb_cache, 0, D * sizeof(float));
    }

    // Pre-compute dense positional encoding grid [D * H * H]
    state.dense_pe_cache.resize(D * H * H);
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < H; ++col) {
            float x_norm = ((float)col + 0.5f) / (float)H;
            float y_norm = ((float)row + 0.5f) / (float)H;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 state.pe_gauss_cache.data(), num_pos_feats);
            for (int d = 0; d < D; ++d)
                state.dense_pe_cache[d + col * D + row * D * H] = pe_vec[d];
        }
    }

    // Pre-compute tiled no-mask embedding [D * H * H]
    state.dense_nomask_cache.resize(D * H * H);
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < H; ++col) {
            for (int d = 0; d < D; ++d)
                state.dense_nomask_cache[d + col * D + row * D * H] = state.no_mask_emb_cache[d];
        }
    }

    state.pe_cache_valid = true;
    fprintf(stderr, "%s: PE cache populated (%d embeddings, %.1f KB dense grids)\n",
            __func__, pe_nel, 2.0f * D * H * H * sizeof(float) / 1024.0f);
}

// Match the official image predictor prompt ordering:
//   box corners (if any) → user points → trailing padding point.
// Even when a box is present, SAM keeps the final padding token because boxes
// are merged into the point stream before calling PromptEncoder(points=..., boxes=None).
static void sam3_collect_pvs_prompt_tokens(const sam3_pvs_params & params,
                                           std::vector<float> & all_coords,
                                           std::vector<int> & all_labels) {
    all_coords.clear();
    all_labels.clear();

    if (params.use_box) {
        all_coords.push_back(params.box.x0);
        all_coords.push_back(params.box.y0);
        all_labels.push_back(2);

        all_coords.push_back(params.box.x1);
        all_coords.push_back(params.box.y1);
        all_labels.push_back(3);
    }

    for (const auto & pt : params.pos_points) {
        all_coords.push_back(pt.x);
        all_coords.push_back(pt.y);
        all_labels.push_back(1);
    }
    for (const auto & pt : params.neg_points) {
        all_coords.push_back(pt.x);
        all_coords.push_back(pt.y);
        all_labels.push_back(0);
    }

    all_coords.push_back(0.0f);
    all_coords.push_back(0.0f);
    all_labels.push_back(-1);
}

// Build sparse and dense embeddings from point/box prompts
// sparse_out: [D, N_pts, 1] where N_pts = n_pos + n_neg + (use_box ? 2 : 0) + pad
// dense_out:  [D, H, H, 1] (no-mask default or mask downsample)
struct sam3_pe_result {
    struct ggml_tensor* sparse;    // [D, N_pts, 1]
    struct ggml_tensor* dense;     // [D, H, H, 1]
    struct ggml_tensor* image_pe;  // [D, H, H, 1] — dense positional encoding grid
    int n_tokens;
};

static sam3_pe_result sam3_build_sam_pe(
        struct ggml_context * ctx,
        const sam3_pvs_params & params,
        int embed_dim, int feat_size)
{
    const int D = embed_dim;  // 256
    const int H = feat_size;  // 72

    // ── Count prompt tokens ──────────────────────────────────────────────
    int N_pts = (int)(params.pos_points.size() + params.neg_points.size()) + 1;  // pad
    if (params.use_box) N_pts += 2;                                             // box corners

    // ── Create input tensors (data uploaded by caller after allocation) ──
    auto * sparse = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_pts, 1);
    ggml_set_name(sparse, "sam_pe_sparse");
    ggml_set_input(sparse);

    auto * image_pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(image_pe, "sam_pe_image_pe");
    ggml_set_input(image_pe);

    auto * dense = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(dense, "sam_pe_dense");
    ggml_set_input(dense);

    sam3_pe_result result;
    result.sparse   = sparse;
    result.dense    = dense;
    result.image_pe = image_pe;
    result.n_tokens = N_pts;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SAM mask decoder — graph building (Phase 6, Step 6.2)
// ═══════════════════════════════════════════════════════════════════════════════

// TwoWayAttentionBlock forward
static void sam3_twoway_block_forward(
    struct ggml_context* ctx,
    struct ggml_tensor*& queries,  // [D, N_q, B] — modified in place
    struct ggml_tensor*& keys,     // [D, N_kv, B] — modified in place
    struct ggml_tensor* query_pe,  // [D, N_q, B]
    struct ggml_tensor* key_pe,    // [D, N_kv, B]
    const sam3_twoway_block& blk,
    int n_heads,
    bool skip_first_layer_pe) {
    // 1. Self-attention on queries
    if (skip_first_layer_pe) {
        // Python: queries = self.self_attn(q=queries, k=queries, v=queries)
        // No residual connection when skipping first layer PE
        queries = sam3_sam_attention(ctx, queries, queries, queries, blk.self_attn, n_heads);
    } else {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, q, queries, blk.self_attn, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
    }
    queries = sam3_layer_norm(ctx, queries, blk.norm1_w, blk.norm1_b);

    // 2. Cross-attention: tokens attending to image
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, blk.ca_tok2img, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, blk.norm2_w, blk.norm2_b);
    }

    // 3. MLP on queries (ReLU activation)
    {
        auto* mlp = ggml_mul_mat(ctx, blk.mlp_fc1_w, queries);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc1_b);
        mlp = ggml_relu(ctx, mlp);
        mlp = ggml_mul_mat(ctx, blk.mlp_fc2_w, mlp);
        mlp = ggml_add(ctx, mlp, blk.mlp_fc2_b);
        queries = ggml_add(ctx, queries, mlp);
        queries = sam3_layer_norm(ctx, queries, blk.norm3_w, blk.norm3_b);
    }

    // 4. Cross-attention: image attending to tokens
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        // Note: q and k are swapped — image (k) attends to tokens (q)
        auto* attn_out = sam3_sam_attention(ctx, k, q, queries, blk.ca_img2tok, n_heads);
        keys = ggml_add(ctx, keys, attn_out);
        keys = sam3_layer_norm(ctx, keys, blk.norm4_w, blk.norm4_b);
    }
}

// MLP forward: N layers with ReLU (except last), optional sigmoid on last
static struct ggml_tensor* sam3_mlp_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* const* weights,
    struct ggml_tensor* const* biases,
    int n_layers,
    bool sigmoid_output = false) {
    for (int i = 0; i < n_layers; ++i) {
        x = ggml_mul_mat(ctx, weights[i], x);
        x = ggml_add(ctx, x, biases[i]);
        if (i < n_layers - 1) {
            x = ggml_relu(ctx, x);
        }
    }
    if (sigmoid_output) {
        x = ggml_sigmoid(ctx, x);
    }
    return x;
}

// Full SAM mask decoder graph
// Inputs:
//   image_feats:  [D, H, H, 1] — tracker neck features (scale 2 = 72×72)
//   image_pe:     [D, H, H, 1] — dense positional encoding
//   sparse_emb:   [D, N_pts, 1] — sparse prompt embeddings
//   dense_emb:    [D, H, H, 1] — dense prompt embeddings (no_mask default)
//   feat_s0:      [D, H0, H0, 1] — high-res features (scale 0 = 288×288)
//   feat_s1:      [D, H1, H1, 1] — mid-res features (scale 1 = 144×144)
// Outputs: sam3_dec_result with masks, iou_pred, obj_score, sam_token_out
struct sam3_dec_result {
    struct ggml_tensor* masks;      // [288*288, N_masks, 1]
    struct ggml_tensor* iou_pred;   // [N_masks, 1]
    struct ggml_tensor* obj_score;  // [1, 1]
    struct ggml_tensor* sam_token;  // [D, 1] — for object pointer
    struct ggml_tensor* mask_tokens;  // [D, N_masks, 1] — raw SAM mask tokens
};

static sam3_dec_result sam3_build_sam_dec_graph(
    struct ggml_context* ctx,
    const sam3_model& model,
    struct ggml_tensor* image_feats,  // [D, H, H, 1]
    struct ggml_tensor* image_pe,     // [D, H, H, 1]
    struct ggml_tensor* sparse_emb,   // [D, N_pts, 1]
    struct ggml_tensor* dense_emb,    // [D, H, H, 1]
    struct ggml_tensor* feat_s0,      // [D, 288, 288, 1] high-res
    struct ggml_tensor* feat_s1)      // [D, 144, 144, 1] mid-res
{
    const auto& dec = model.sam_dec;
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;  // 256
    const int H = hp.n_img_embd();   // 72
    const int N_pts = (int)sparse_emb->ne[1];
    const int n_heads = 8;                               // SAM uses 8 heads
    const int num_mask_tokens = hp.sam_n_multimask + 1;  // 4

    // ── Concatenate output tokens ────────────────────────────────────────
    // Tokens: [obj_score_token(1, D), iou_token(1, D), mask_tokens(4, D)]
    // Total special tokens: 1 + 1 + 4 = 6
    // Then concatenate with sparse prompt embeddings
    // Result: [D, 6 + N_pts, 1]

    // obj_score_token: [D, 1] → use directly
    // iou_token: [D, 1] → use directly
    // mask_tokens: [D, 4] → need to join

    // Concatenate: obj_score_token, iou_token, mask_tokens → [D, 6]
    auto* output_tokens = ggml_concat(ctx, dec.obj_score_token, dec.iou_token, 1);  // [D, 2]
    output_tokens = ggml_concat(ctx, output_tokens, dec.mask_tokens, 1);            // [D, 6]
    // Add batch dim: [D, 6, 1]
    output_tokens = ggml_reshape_3d(ctx, output_tokens, D, 6, 1);

    // sparse_emb is already [D, N_pts, 1]
    // tokens = cat(output_tokens, sparse_emb) along dim 1 → [D, 6+N_pts, 1]
    auto* tokens = ggml_concat(ctx, output_tokens, sparse_emb, 1);
    ggml_set_name(tokens, "sam_dec_tokens_initial");

    const int N_tok = 6 + N_pts;

    // ── Prepare image src and pos ────────────────────────────────────────
    // src = image_feats + dense_emb  [D, H, H, 1]
    auto* src = ggml_add(ctx, image_feats, dense_emb);

    // Flatten spatial: [D, H*H, 1]
    src = ggml_reshape_3d(ctx, src, D, H * H, 1);
    auto* pos_src = ggml_reshape_3d(ctx, image_pe, D, H * H, 1);

    // ── TwoWay transformer blocks ────────────────────────────────────────
    auto* queries = tokens;   // [D, N_tok, 1]
    auto* keys = src;         // [D, H*H, 1]
    auto* query_pe = tokens;  // query PE = initial point embedding (same as tokens)
    auto* key_pe = pos_src;

    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        sam3_twoway_block_forward(ctx, queries, keys, query_pe, key_pe,
                                  dec.twoway_blocks[i], n_heads,
                                  /*skip_first_layer_pe=*/(i == 0));
        sam3_name_tensorf(queries, "sam_dec_block%d_queries", i);
        sam3_name_tensorf(keys, "sam_dec_block%d_keys", i);
    }

    // Final attention: tokens → image
    {
        auto* q = ggml_add(ctx, queries, query_pe);
        auto* k = ggml_add(ctx, keys, key_pe);
        auto* attn_out = sam3_sam_attention(ctx, q, k, keys, dec.final_attn, n_heads);
        queries = ggml_add(ctx, queries, attn_out);
        queries = sam3_layer_norm(ctx, queries, dec.final_norm_w, dec.final_norm_b);
        ggml_set_name(queries, "sam_dec_final_queries");
    }

    // ── Extract output tokens ────────────────────────────────────────────
    // queries: [D, N_tok, 1]
    // s=1 (obj_score at index 0)
    // iou_token_out = queries[:, 1, :]
    // mask_tokens_out = queries[:, 2:6, :]
    // obj_score in = queries[:, 0, :]

    auto* iou_token_out = ggml_view_3d(ctx, queries, D, 1, 1,
                                       queries->nb[1], queries->nb[2],
                                       1 * queries->nb[1]);
    iou_token_out = ggml_cont(ctx, iou_token_out);  // [D, 1, 1]

    auto* mask_tokens_out = ggml_view_3d(ctx, queries, D, num_mask_tokens, 1,
                                         queries->nb[1], queries->nb[2],
                                         2 * queries->nb[1]);
    mask_tokens_out = ggml_cont(ctx, mask_tokens_out);  // [D, 4, 1]
    ggml_set_name(mask_tokens_out, "sam_dec_mask_tokens");

    auto* obj_in = ggml_view_3d(ctx, queries, D, 1, 1,
                                queries->nb[1], queries->nb[2], 0);
    obj_in = ggml_cont(ctx, obj_in);  // [D, 1, 1]

    // Also extract SAM output token (index 2 = first mask token, used for object pointer)
    auto* sam_token = ggml_view_2d(ctx, queries, D, 1,
                                   queries->nb[1], 2 * queries->nb[1]);
    sam_token = ggml_cont(ctx, sam_token);  // [D, 1]
    ggml_set_name(sam_token, "sam_dec_sam_token");

    // ── Upscale mask embeddings ──────────────────────────────────────────
    // src (keys after transformer): [D, H*H, 1] → [D, H, H, 1]
    // Neck output layout is [C, W, H, B] throughout.
    auto * src_img = ggml_reshape_4d(ctx, keys, D, H, H, 1);

    // Permute [C, W, H, B] → [W, H, C, B] for conv_transpose_2d
    src_img = ggml_cont(ctx, ggml_permute(ctx, src_img, 2, 0, 1, 3));  // [H, H, D, 1]

    // dc1: ConvTranspose2d(256, 64, k=2, s=2) → output [W, H, C, B] = [144, 144, 64, 1]
    auto * up1 = ggml_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, dec.up1_w), src_img, 2);
    up1 = ggml_add(ctx, up1, ggml_reshape_4d(ctx, dec.up1_b, 1, 1, ggml_nelements(dec.up1_b), 1));

    // conv_s1: 1x1 conv on feat_s1 (256→64). feat_s1 is [C, W, H, B] — permute for conv.
    auto * fs1 = ggml_cont(ctx, ggml_permute(ctx, feat_s1, 2, 0, 1, 3));  // [W, H, C, B]
    auto * hs1 = ggml_conv_2d_sk_p0(ctx, dec.conv_s1_w, fs1);  // [W, H, 64, B]
    hs1 = ggml_add(ctx, hs1, ggml_reshape_4d(ctx, dec.conv_s1_b, 1, 1, 64, 1));
    ggml_set_name(hs1, "sam_dec_feat_s1_proj");

    // Python: act1(ln1(dc1(src) + feat_s1)) — add feat_s1 BEFORE LayerNorm
    up1 = ggml_add(ctx, up1, hs1);  // both [W, H, 64, B]

    // Permute to [C, W, H, B] for LayerNorm2d
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 1, 2, 0, 3));  // [64, 144, 144, 1]
    up1 = sam3_layer_norm_2d(ctx, up1, dec.up1_norm_w, dec.up1_norm_b);

    // GELU activation (exact, matching Python nn.GELU)
    up1 = ggml_gelu_erf(ctx, up1);  // still [C, W, H, B]

    // Permute back to [W, H, C, B] for next deconv
    up1 = ggml_cont(ctx, ggml_permute(ctx, up1, 2, 0, 1, 3));  // [144, 144, 64, 1]

    // dc2: ConvTranspose2d(64, 32, k=2, s=2) → [288, 288, 32, 1]
    auto * up2 = ggml_conv_transpose_2d_p0(ctx, sam3_conv_transpose_weight(ctx, dec.up2_w), up1, 2);
    up2 = ggml_add(ctx, up2, ggml_reshape_4d(ctx, dec.up2_b, 1, 1, ggml_nelements(dec.up2_b), 1));

    // conv_s0: 1x1 conv on feat_s0 (256→32). feat_s0 is [C, W, H, B] — permute for conv.
    auto * fs0 = ggml_cont(ctx, ggml_permute(ctx, feat_s0, 2, 0, 1, 3));  // [W, H, C, B]
    auto * hs0 = ggml_conv_2d_sk_p0(ctx, dec.conv_s0_w, fs0);  // [W, H, 32, B]
    hs0 = ggml_add(ctx, hs0, ggml_reshape_4d(ctx, dec.conv_s0_b, 1, 1, 32, 1));
    ggml_set_name(hs0, "sam_dec_feat_s0_proj");

    // Python: act2(dc2(upscaled_embedding) + feat_s0) — no LayerNorm here
    up2 = ggml_add(ctx, up2, hs0);  // both [W, H, 32, B]

    // Permute to [C, W, H, B] for subsequent operations
    up2 = ggml_cont(ctx, ggml_permute(ctx, up2, 1, 2, 0, 3));  // [32, 288, 288, 1]

    // GELU activation (exact, matching Python nn.GELU)
    up2 = ggml_gelu_erf(ctx, up2);

    // up2: [32, 288, 288, 1] — this is our upscaled_embedding
    ggml_set_name(up2, "sam_dec_upscaled");

    // ── Hypernetwork: predict masks ──────────────────────────────────────
    // For each mask token i, pass through 3-layer MLP to get [32] vector
    // Then dot product with upscaled_embedding [32, 288*288] to get mask
    // Flatten upscaled: [32, 288*288, 1]
    auto* up_flat = ggml_reshape_3d(ctx, up2, 32, 288 * 288, 1);  // [32, 288*288, 1]

    // Process each mask token through its hypernetwork MLP
    // mask_tokens_out: [D, 4, 1]
    struct ggml_tensor* mask_list[4];
    for (int m = 0; m < num_mask_tokens; ++m) {
        // Extract token m: [D, 1, 1]
        auto* tok = ggml_view_3d(ctx, mask_tokens_out, D, 1, 1,
                                 mask_tokens_out->nb[1], mask_tokens_out->nb[2],
                                 m * mask_tokens_out->nb[1]);
        tok = ggml_cont(ctx, tok);  // [D, 1, 1]

        // MLP: 3 layers, 256→256→256→32, ReLU on first two
        auto* hyper = sam3_mlp_forward(ctx, tok,
                                       dec.hyper_w[m], dec.hyper_b[m], 3);
        // hyper: [32, 1, 1]

        // Dot product: hyper^T @ up_flat → [1, 288*288, 1]
        // Use mul_mat: up_flat^T [288*288, 32] @ hyper [32, 1] → [288*288, 1, 1]
        auto* mask = ggml_mul_mat(ctx, up_flat, hyper);  // [288*288, 1, 1]
        mask_list[m] = mask;
    }

    // Stack masks: [288*288, 4, 1]
    auto* masks = mask_list[0];
    for (int m = 1; m < num_mask_tokens; ++m) {
        masks = ggml_concat(ctx, masks, mask_list[m], 1);
    }
    ggml_set_name(masks, "sam_dec_masks");

    // ── IoU prediction ───────────────────────────────────────────────────
    // iou_token_out: [D, 1, 1]
    auto* iou_pred = sam3_mlp_forward(ctx, iou_token_out,
                                      dec.iou_head_w, dec.iou_head_b, 3,
                                      /*sigmoid_output=*/true);
    // iou_pred: [4, 1, 1] → reshape to [4, 1]
    iou_pred = ggml_reshape_2d(ctx, iou_pred, num_mask_tokens, 1);
    ggml_set_name(iou_pred, "sam_dec_iou");

    // ── Object score ─────────────────────────────────────────────────────
    // obj_in: [D, 1, 1]
    auto* obj_score = sam3_mlp_forward(ctx, obj_in,
                                       dec.obj_head_w, dec.obj_head_b, 3);
    // obj_score: [1, 1, 1] → reshape to [1, 1]
    obj_score = ggml_reshape_2d(ctx, obj_score, 1, 1);
    ggml_set_name(obj_score, "sam_dec_obj_score");

    sam3_dec_result res;
    res.masks = masks;
    res.iou_pred = iou_pred;
    res.obj_score = obj_score;
    res.sam_token = sam_token;
    res.mask_tokens = mask_tokens_out;
    return res;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image segmentation — PVS (visual-prompted) (Phase 6, Step 6.3)
// ═══════════════════════════════════════════════════════════════════════════════

sam3_result sam3_segment_pvs(sam3_state& state,
                             const sam3_model& model,
                             const sam3_pvs_params& params) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;                      // 256
    const int H = hp.n_img_embd();                       // 72
    const int num_mask_tokens = hp.sam_n_multimask + 1;  // 4
    sam3_result result;

    // ── Validate ─────────────────────────────────────────────────────────
    if (!state.neck_trk[0]) {
        fprintf(stderr, "%s: image not encoded — call sam3_encode_image first\n", __func__);
        return result;
    }
    if (params.pos_points.empty() && !params.use_box) {
        fprintf(stderr, "%s: no prompts provided (need at least one point or box)\n", __func__);
        return result;
    }

    fprintf(stderr, "%s: %zu pos points, %zu neg points, box=%s, multimask=%s\n",
            __func__, params.pos_points.size(), params.neg_points.size(),
            params.use_box ? "yes" : "no", params.multimask ? "yes" : "no");

    // ── Build computation graph ──────────────────────────────────────────
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context* ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return result;
    }

    // ── SAM prompt encoder (CPU pre-compute + input tensors) ─────────────
    auto pe_out = sam3_build_sam_pe(ctx0, params, D, H);

    // Match the Python image predictor: add the learned no-memory token to the
    // lowest-resolution tracker feature map before the SAM decoder.
    auto * no_mem = ggml_reshape_4d(ctx0, model.tensors.at("no_mem_embed"), D, 1, 1, 1);
    auto * image_feats = ggml_add(ctx0, state.neck_trk[2], no_mem);
    ggml_set_name(image_feats, "sam_dec_image_feats");

    // ── SAM mask decoder graph ───────────────────────────────────────────
    auto dec_out = sam3_build_sam_dec_graph(ctx0, model,
                                            image_feats,        // [D, 72, 72, 1]
                                            pe_out.image_pe,
                                            pe_out.sparse,
                                            pe_out.dense,
                                            state.neck_trk[0],   // [D, 288, 288, 1]
                                            state.neck_trk[1]);  // [D, 144, 144, 1]

    // Mark outputs
    ggml_set_output(dec_out.masks);
    ggml_set_output(dec_out.iou_pred);
    ggml_set_output(dec_out.obj_score);
    ggml_set_output(dec_out.sam_token);

    // ── Build and allocate graph ─────────────────────────────────────────
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx0, 32768, false);
    ggml_build_forward_expand(graph, dec_out.masks);
    ggml_build_forward_expand(graph, dec_out.iou_pred);
    ggml_build_forward_expand(graph, dec_out.obj_score);
    ggml_build_forward_expand(graph, dec_out.sam_token);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph)) {
        fprintf(stderr, "%s: failed to reserve graph memory\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    }

    fprintf(stderr, "%s: graph allocated, %d nodes\n", __func__, ggml_graph_n_nodes(graph));

    // ── Upload input data (using cached embeddings) ────────────────────
    // Populate PE cache on first call (reads model weights from GPU once)
    sam3_populate_pe_cache(state, model);

    {
        const int N_pts = pe_out.n_tokens;
        const int num_pos_feats = D / 2;

        std::vector<float> all_coords;
        std::vector<int> all_labels;
        sam3_collect_pvs_prompt_tokens(params, all_coords, all_labels);

        // Sparse embeddings — only this changes per call
        std::vector<float> sparse_data(N_pts * D, 0.0f);
        for (int p = 0; p < N_pts; ++p) {
            float px = all_coords[p * 2 + 0] + 0.5f;
            float py = all_coords[p * 2 + 1] + 0.5f;
            float x_norm = px / (float)hp.img_size;
            float y_norm = py / (float)hp.img_size;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 state.pe_gauss_cache.data(), num_pos_feats);
            int label = all_labels[p];
            if (label == -1) {
                for (int d = 0; d < D; ++d)
                    sparse_data[p * D + d] = state.not_a_point_cache[d];
            } else {
                for (int d = 0; d < D; ++d)
                    sparse_data[p * D + d] = pe_vec[d] + state.point_emb_cache[label][d];
            }
        }
        ggml_backend_tensor_set(pe_out.sparse, sparse_data.data(), 0, N_pts * D * sizeof(float));

        // Dense PE grid and no-mask embedding — use pre-computed caches
        ggml_backend_tensor_set(pe_out.image_pe, state.dense_pe_cache.data(),
                                0, D * H * H * sizeof(float));
        ggml_backend_tensor_set(pe_out.dense, state.dense_nomask_cache.data(),
                                0, D * H * H * sizeof(float));
    }

    // ── Compute ──────────────────────────────────────────────────────────
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        sam3_graph_compute(model.backend, graph, state.n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "%s: graph computed in %.1f ms (%d threads)\n",
                __func__, ms, state.n_threads);
    }

    // ── Read outputs ─────────────────────────────────────────────────────
    // masks: [288*288, 4, 1]
    const int mask_hw = 288;
    std::vector<float> masks_data(mask_hw * mask_hw * num_mask_tokens);
    ggml_backend_tensor_get(dec_out.masks, masks_data.data(), 0, masks_data.size() * sizeof(float));

    // IoU predictions: [4, 1]
    std::vector<float> iou_data(num_mask_tokens);
    ggml_backend_tensor_get(dec_out.iou_pred, iou_data.data(), 0, num_mask_tokens * sizeof(float));

    // Object score: [1, 1]
    float obj_logit = 0.0f;
    ggml_backend_tensor_get(dec_out.obj_score, &obj_logit, 0, sizeof(float));
    float obj_score = 1.0f / (1.0f + expf(-obj_logit));

    fprintf(stderr, "%s: obj_score=%.4f (logit=%.4f), iou=[%.3f, %.3f, %.3f, %.3f]\n",
            __func__, obj_score, obj_logit,
            iou_data[0], iou_data[1], iou_data[2], iou_data[3]);

    // ── Select masks based on multimask mode ─────────────────────────────
    // Python: if multimask_output → masks[:, 1:, :, :], iou_pred[:, 1:]
    //         else                → masks[:, 0:1, :, :], iou_pred[:, 0:1]
    int start_idx, end_idx;
    if (params.multimask) {
        start_idx = 1;
        end_idx = num_mask_tokens;
    } else {
        start_idx = 0;
        end_idx = 1;
    }

    for (int m = start_idx; m < end_idx; ++m) {
        sam3_detection det;

        // Resize mask from 288×288 to original image size
        const float* mask_ptr = masks_data.data() + m * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);

        // Binarize at threshold 0.0 (sigmoid(logit) > 0.5 ↔ logit > 0.0)
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i) {
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        }

        det.mask.iou_score = iou_data[m];
        det.mask.obj_score = obj_score;
        det.mask.instance_id = m;
        det.score = iou_data[m];
        det.iou_score = iou_data[m];
        det.instance_id = m;

        // Compute bounding box from mask
        int min_x = state.orig_width, min_y = state.orig_height;
        int max_x = 0, max_y = 0;
        for (int y = 0; y < state.orig_height; ++y) {
            for (int x = 0; x < state.orig_width; ++x) {
                if (det.mask.data[y * state.orig_width + x] > 0) {
                    min_x = std::min(min_x, x);
                    min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x);
                    max_y = std::max(max_y, y);
                }
            }
        }
        det.box = {(float)min_x, (float)min_y, (float)max_x, (float)max_y};

        result.detections.push_back(std::move(det));
    }

    fprintf(stderr, "%s: %zu masks returned\n", __func__, result.detections.size());

    // ── Cleanup ──────────────────────────────────────────────────────────
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    fprintf(stderr, "%s: completed in %.1f ms\n", __func__, total_ms);

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Video tracking (Phase 7)
// ═══════════════════════════════════════════════════════════════════════════════

struct sam3_prop_output {
    std::vector<float> mask_logits;
    std::vector<float> iou_scores;
    float              obj_score;
    std::vector<float> sam_token;
    int n_masks, mask_h, mask_w;
};

static sam3_prop_output sam3_propagate_single(
    sam3_state& state, const sam3_model& model,
    const sam3_masklet& masklet,
    const std::vector<sam3_memory_slot>& mem_bank,
    const std::vector<std::pair<int, struct ggml_tensor*>>& ptr_bank)
{
    sam3_prop_output output = {};
    const auto& hp = model.hparams;
    const int D = hp.neck_dim, MD = hp.mem_out_dim, H = hp.n_img_embd();

    auto sel = sam3_select_memory_frames(mem_bank, hp.num_maskmem);
    if (sel.empty()) return output;

    int n_sel = (int)sel.size(), M = n_sel * H * H;
    std::vector<float> mem_data(MD * M);
    for (int s = 0; s < n_sel; ++s)
        ggml_backend_tensor_get(mem_bank[sel[s]].spatial_feats,
                                 mem_data.data() + s * MD * H * H, 0, MD * H * H * sizeof(float));

    int P = std::min((int)ptr_bank.size(), hp.max_obj_ptrs);
    std::vector<float> ptr_data(D * P);
    for (int p = 0; p < P; ++p)
        ggml_backend_tensor_get(ptr_bank[p].second, ptr_data.data() + p * D, 0, D * sizeof(float));

    const size_t buf_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {buf_size, nullptr, true};
    auto* ctx0 = ggml_init(gparams);
    if (!ctx0) return output;

    auto* curr = ggml_reshape_3d(ctx0, state.neck_trk[2], D, H * H, 1);
    auto* mem_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, MD, M, 1);
    ggml_set_name(mem_in, "mem_feats"); ggml_set_input(mem_in);
    struct ggml_tensor* ptr_in = nullptr;
    if (P > 0) { ptr_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, P, 1);
                  ggml_set_name(ptr_in, "obj_ptrs"); ggml_set_input(ptr_in); }

    auto* conditioned = sam3_build_mem_attn_graph(ctx0, model, curr, mem_in, ptr_in);
    auto* cond_spatial = ggml_reshape_4d(ctx0, conditioned, D, H, H, 1);

    auto* empty_sparse = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, 0, 1);
    ggml_set_name(empty_sparse, "prop_sparse");
    auto* image_pe = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(image_pe, "prop_pe"); ggml_set_input(image_pe);
    auto* dense_emb = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H, H, 1);
    ggml_set_name(dense_emb, "prop_dense"); ggml_set_input(dense_emb);

    auto dec = sam3_build_sam_dec_graph(ctx0, model, cond_spatial, image_pe,
                                         empty_sparse, dense_emb,
                                         state.neck_trk[0], state.neck_trk[1]);
    ggml_set_output(dec.masks); ggml_set_output(dec.iou_pred);
    ggml_set_output(dec.obj_score); ggml_set_output(dec.sam_token);

    auto* graph = ggml_new_graph_custom(ctx0, 32768, false);
    ggml_build_forward_expand(graph, dec.masks);
    ggml_build_forward_expand(graph, dec.iou_pred);
    ggml_build_forward_expand(graph, dec.obj_score);
    ggml_build_forward_expand(graph, dec.sam_token);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_gallocr_free(galloc); ggml_free(ctx0); return output; }

    ggml_backend_tensor_set(mem_in, mem_data.data(), 0, mem_data.size() * sizeof(float));
    if (ptr_in && P > 0)
        ggml_backend_tensor_set(ptr_in, ptr_data.data(), 0, ptr_data.size() * sizeof(float));

    // Upload image_pe and dense_emb
    {
        const int npf = D / 2;
        std::vector<float> pe_g(D);
        if (model.sam_pe.pe_gaussian->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(D);
            ggml_backend_tensor_get(model.sam_pe.pe_gaussian, tmp.data(), 0, D * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), pe_g.data(), D);
        } else ggml_backend_tensor_get(model.sam_pe.pe_gaussian, pe_g.data(), 0, D * sizeof(float));
        std::vector<float> pe_d(D * H * H);
        for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c) {
            float xn = ((float)c + 0.5f) / H, yn = ((float)r + 0.5f) / H;
            float pv[256]; sam3_pe_encode_coord(pv, xn, yn, pe_g.data(), npf);
            for (int d = 0; d < D; ++d) pe_d[d + c*D + r*D*H] = pv[d];
        }
        ggml_backend_tensor_set(image_pe, pe_d.data(), 0, pe_d.size() * sizeof(float));

        float nm[256];
        if (model.sam_pe.no_mask_embed->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(D);
            ggml_backend_tensor_get(model.sam_pe.no_mask_embed, tmp.data(), 0, D * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), nm, D);
        } else ggml_backend_tensor_get(model.sam_pe.no_mask_embed, nm, 0, D * sizeof(float));
        std::vector<float> dd(D * H * H);
        for (int r = 0; r < H; ++r) for (int c = 0; c < H; ++c)
            for (int d = 0; d < D; ++d) dd[d + c*D + r*D*H] = nm[d];
        ggml_backend_tensor_set(dense_emb, dd.data(), 0, dd.size() * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, 4);

    const int mhw = 288;
    output.n_masks = 1; output.mask_h = mhw; output.mask_w = mhw;
    output.mask_logits.resize(mhw * mhw);
    ggml_backend_tensor_get(dec.masks, output.mask_logits.data(), 0, mhw*mhw*sizeof(float));
    output.iou_scores.resize(1);
    ggml_backend_tensor_get(dec.iou_pred, output.iou_scores.data(), 0, sizeof(float));
    ggml_backend_tensor_get(dec.obj_score, &output.obj_score, 0, sizeof(float));
    output.sam_token.resize(D);
    ggml_backend_tensor_get(dec.sam_token, output.sam_token.data(), 0, D * sizeof(float));

    ggml_gallocr_free(galloc); ggml_free(ctx0);
    return output;
}

static std::vector<std::pair<int,int>> sam3_match_detections(
    const std::vector<sam3_masklet>& masklets, const std::vector<sam3_detection>& dets,
    const std::vector<sam3_mask>& prop_masks, float iou_threshold) {
    std::vector<std::pair<int,int>> matches;
    if (masklets.empty() || dets.empty()) return matches;
    int n_m = (int)masklets.size(), n_d = (int)dets.size();
    std::vector<bool> dm(n_d, false);
    for (int i = 0; i < n_m; ++i) {
        if (i >= (int)prop_masks.size() || prop_masks[i].data.empty()) continue;
        int bj = -1; float bi = iou_threshold;
        for (int j = 0; j < n_d; ++j) {
            if (dm[j] || dets[j].mask.data.empty()) continue;
            int w = prop_masks[i].width, h = prop_masks[i].height;
            if (w != dets[j].mask.width || h != dets[j].mask.height) continue;
            float iou = sam3_mask_iou(prop_masks[i].data.data(), dets[j].mask.data.data(), w*h);
            if (iou > bi) { bi = iou; bj = j; }
        }
        if (bj >= 0) { matches.push_back({i, bj}); dm[bj] = true; }
    }
    return matches;
}

static void sam3_update_tracker(sam3_tracker& tracker, int frame_idx) {
    for (auto it = tracker.pending.begin(); it != tracker.pending.end(); ) {
        int age = frame_idx - it->first_frame;
        if (age >= tracker.params.hotstart_delay && it->mds_sum > 0) {
            it->confirmed = true; tracker.masklets.push_back(std::move(*it));
            it = tracker.pending.erase(it);
        } else if (age >= tracker.params.hotstart_delay) {
            it = tracker.pending.erase(it);
        } else ++it;
    }
    for (auto it = tracker.masklets.begin(); it != tracker.masklets.end(); ) {
        if (frame_idx - it->last_seen > tracker.params.max_keep_alive) {
            tracker.mem_banks.erase(it->instance_id);
            tracker.ptr_banks.erase(it->instance_id);
            it = tracker.masklets.erase(it);
        } else ++it;
    }
}

static bool sam3_encode_memory(
    sam3_tracker& tracker, sam3_state& state, const sam3_model& model,
    int inst_id, const float* mask_logits, int mask_h, int mask_w,
    int frame_idx, bool is_cond) {
    const auto& hp = model.hparams;
    const int D = hp.neck_dim, MD = hp.mem_out_dim, H = hp.n_img_embd();
    const size_t bs = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params gp = {bs, nullptr, true};
    auto* ctx0 = ggml_init(gp); if (!ctx0) return false;

    auto rm = sam3_bilinear_interpolate(mask_logits, mask_w, mask_h, H, H);
    for (auto& v : rm) v = 1.0f / (1.0f + expf(-v));

    auto* mi = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, H, H, 1);
    ggml_set_name(mi, "mem_mask"); ggml_set_input(mi);

    auto* pix = ggml_conv_2d(ctx0, model.mem_enc.pix_proj_w, state.neck_trk[2], 1,1,0,0,1,1);
    pix = ggml_add(ctx0, pix, ggml_reshape_4d(ctx0, model.mem_enc.pix_proj_b, 1,1,D,1));
    auto* fused = ggml_mul(ctx0, pix, mi);
    for (int i = 0; i < 2; ++i)
        fused = sam3_cxblock_forward(ctx0, fused,
            model.mem_enc.fuser_dw_w[i], model.mem_enc.fuser_dw_b[i],
            model.mem_enc.fuser_norm_w[i], model.mem_enc.fuser_norm_b[i],
            model.mem_enc.fuser_fc1_w[i], model.mem_enc.fuser_fc1_b[i],
            model.mem_enc.fuser_fc2_w[i], model.mem_enc.fuser_fc2_b[i],
            model.mem_enc.fuser_gamma[i]);
    auto* mo = ggml_conv_2d(ctx0, model.mem_enc.out_proj_w, fused, 1,1,0,0,1,1);
    mo = ggml_add(ctx0, mo, ggml_reshape_4d(ctx0, model.mem_enc.out_proj_b, 1,1,MD,1));
    ggml_set_name(mo, "mem_out"); ggml_set_output(mo);

    auto* g = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(g, mo);
    auto* ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(ga, g) || !ggml_gallocr_alloc_graph(ga, g)) {
        ggml_gallocr_free(ga); ggml_free(ctx0); return false; }
    ggml_backend_tensor_set(mi, rm.data(), 0, rm.size() * sizeof(float));
    sam3_graph_compute(model.backend, g, 4);

    std::vector<float> md(MD * H * H);
    ggml_backend_tensor_get(mo, md.data(), 0, md.size() * sizeof(float));

    if (!tracker.ctx) { struct ggml_init_params tp = {ggml_tensor_overhead()*4096,nullptr,true};
                        tracker.ctx = ggml_init(tp); }
    auto* st = ggml_new_tensor_4d(tracker.ctx, GGML_TYPE_F32, MD, H, H, 1);
    auto* sb = ggml_backend_alloc_buffer(model.backend, MD*H*H*sizeof(float));
    struct ggml_tallocr ta = ggml_tallocr_new(sb);
    ggml_tallocr_alloc(&ta, st); tracker.owned_buffers.push_back(sb);
    ggml_backend_tensor_set(st, md.data(), 0, md.size() * sizeof(float));

    sam3_memory_slot slot; slot.spatial_feats = st;
    slot.frame_index = frame_idx; slot.is_cond_frame = is_cond;
    auto& bk = tracker.mem_banks[inst_id]; bk.push_back(slot);
    while ((int)bk.size() > hp.num_maskmem) {
        bool removed = false;
        for (auto it = bk.begin(); it != bk.end(); ++it)
            if (!it->is_cond_frame) { bk.erase(it); removed = true; break; }
        if (!removed) bk.erase(bk.begin() + 1);
    }
    ggml_gallocr_free(ga); ggml_free(ctx0); return true;
}

static void sam3_store_obj_ptr(
    sam3_tracker& tracker, const sam3_model& model,
    int inst_id, const float* pd, int frame_idx) {
    const int D = model.hparams.neck_dim;
    if (!tracker.ctx) { struct ggml_init_params tp = {ggml_tensor_overhead()*4096,nullptr,true};
                        tracker.ctx = ggml_init(tp); }
    auto* pt = ggml_new_tensor_2d(tracker.ctx, GGML_TYPE_F32, D, 1);
    auto* pb = ggml_backend_alloc_buffer(model.backend, D * sizeof(float));
    struct ggml_tallocr ta = ggml_tallocr_new(pb);
    ggml_tallocr_alloc(&ta, pt); tracker.owned_buffers.push_back(pb);
    ggml_backend_tensor_set(pt, pd, 0, D * sizeof(float));
    auto& bk = tracker.ptr_banks[inst_id]; bk.push_back({frame_idx, pt});
    while ((int)bk.size() > model.hparams.max_obj_ptrs) bk.erase(bk.begin());
}

sam3_tracker_ptr sam3_create_tracker(const sam3_model& model,
                                     const sam3_video_params& params) {
    sam3_tracker_ptr tracker(new sam3_tracker());
    tracker->params = params;
    fprintf(stderr, "%s: tracker created (hotstart=%d, max_keep_alive=%d)\n",
            __func__, params.hotstart_delay, params.max_keep_alive);
    return tracker;
}

sam3_result sam3_track_frame(sam3_tracker& tracker, sam3_state& state,
                             const sam3_model& model, const sam3_image& frame) {
    sam3_result result;
    const int D = model.hparams.neck_dim;
    if (!sam3_encode_image(state, model, frame)) return result;
    int fi = tracker.frame_index;
    fprintf(stderr, "%s: frame %d (%zu active + %zu pending)\n",
            __func__, fi, tracker.masklets.size(), tracker.pending.size());

    std::map<int, sam3_mask> pm; std::map<int, sam3_prop_output> po;
    for (auto& ml : tracker.masklets) {
        int id = ml.instance_id;
        auto im = tracker.mem_banks.find(id);
        if (im == tracker.mem_banks.end() || im->second.empty()) continue;
        po[id] = sam3_propagate_single(state, model, ml, im->second, tracker.ptr_banks[id]);
        if (po[id].mask_logits.empty()) continue;
        auto rs = sam3_bilinear_interpolate(po[id].mask_logits.data(),
                    po[id].mask_w, po[id].mask_h, state.orig_width, state.orig_height);
        pm[id].width = state.orig_width; pm[id].height = state.orig_height;
        pm[id].data.resize(state.orig_width * state.orig_height);
        int fg = 0;
        for (int p = 0; p < (int)rs.size(); ++p) {
            bool f = rs[p] > 0.0f; pm[id].data[p] = f ? 255 : 0; if (f) fg++;
        }
        ml.last_score = po[id].iou_scores[0]; ml.last_seen = fi;
        float cov = (float)fg / (state.orig_width * state.orig_height);
        ml.mds_sum += (cov > 0.001f && po[id].obj_score > 0.0f) ? 1 : -1;
    }
    for (auto& ml : tracker.pending) {
        int id = ml.instance_id;
        auto im = tracker.mem_banks.find(id);
        if (im == tracker.mem_banks.end() || im->second.empty()) continue;
        auto p2 = sam3_propagate_single(state, model, ml, im->second, tracker.ptr_banks[id]);
        if (!p2.mask_logits.empty()) {
            ml.last_score = p2.iou_scores[0]; ml.last_seen = fi;
            auto r2 = sam3_bilinear_interpolate(p2.mask_logits.data(),
                        p2.mask_w, p2.mask_h, state.orig_width, state.orig_height);
            int fg2 = 0; for (auto v : r2) if (v > 0.0f) fg2++;
            float c2 = (float)fg2 / (state.orig_width * state.orig_height);
            ml.mds_sum += (c2 > 0.001f && p2.obj_score > 0.0f) ? 1 : -1;
            sam3_encode_memory(tracker, state, model, id,
                                p2.mask_logits.data(), p2.mask_h, p2.mask_w, fi, false);
            std::vector<float> op(D);
            sam3_extract_obj_ptr_cpu(model, p2.sam_token.data(), p2.obj_score, op.data());
            sam3_store_obj_ptr(tracker, model, id, op.data(), fi);
        }
    }
    sam3_result nd;
    if (!tracker.params.text_prompt.empty()) {
        sam3_pcs_params pcs; pcs.text_prompt = tracker.params.text_prompt;
        pcs.score_threshold = tracker.params.score_threshold;
        pcs.nms_threshold = tracker.params.nms_threshold;
        nd = sam3_segment_pcs(state, model, pcs);
    }
    std::vector<sam3_mask> pmv(tracker.masklets.size());
    for (int i = 0; i < (int)tracker.masklets.size(); ++i) {
        auto it = pm.find(tracker.masklets[i].instance_id);
        if (it != pm.end()) pmv[i] = it->second;
    }
    auto mat = sam3_match_detections(tracker.masklets, nd.detections,
                                      pmv, tracker.params.assoc_iou_threshold);
    std::vector<bool> dmat(nd.detections.size(), false);
    for (auto& m : mat) { dmat[m.second] = true; tracker.masklets[m.first].last_seen = fi; }
    for (int j = 0; j < (int)nd.detections.size(); ++j) {
        if (dmat[j]) continue;
        sam3_masklet ml; ml.instance_id = tracker.next_inst_id++;
        ml.first_frame = fi; ml.last_seen = fi;
        ml.last_score = nd.detections[j].score; ml.mds_sum = 1;
        tracker.pending.push_back(std::move(ml));
    }
    for (auto& ml : tracker.masklets) {
        int id = ml.instance_id;
        auto it = po.find(id);
        if (it == po.end() || it->second.mask_logits.empty()) continue;
        sam3_encode_memory(tracker, state, model, id,
                            it->second.mask_logits.data(), it->second.mask_h, it->second.mask_w, fi, false);
        std::vector<float> op(D);
        sam3_extract_obj_ptr_cpu(model, it->second.sam_token.data(), it->second.obj_score, op.data());
        sam3_store_obj_ptr(tracker, model, id, op.data(), fi);
    }
    sam3_update_tracker(tracker, fi);
    for (auto& ml : tracker.masklets) {
        auto it = pm.find(ml.instance_id);
        if (it == pm.end() || it->second.data.empty()) continue;
        sam3_detection det; det.instance_id = ml.instance_id;
        det.score = ml.last_score; det.mask = it->second;
        det.mask.instance_id = ml.instance_id; det.mask.iou_score = ml.last_score;
        float x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f;
        for (int p = 0; p < (int)det.mask.data.size(); ++p) if (det.mask.data[p]>127) {
            int x=p%det.mask.width, y=p/det.mask.width;
            x0=std::min(x0,(float)x); y0=std::min(y0,(float)y);
            x1=std::max(x1,(float)x); y1=std::max(y1,(float)y);
        }
        if (x0<=x1) det.box={x0,y0,x1,y1};
        result.detections.push_back(std::move(det));
    }
    sam3_resolve_overlaps(result.detections);
    for (auto& d : result.detections) { if (d.mask.data.empty()) continue;
        sam3_fill_holes(d.mask.data.data(), d.mask.width, d.mask.height, tracker.params.fill_hole_area);
        sam3_remove_sprinkles(d.mask.data.data(), d.mask.width, d.mask.height, tracker.params.fill_hole_area);
    }
    tracker.frame_index++;
    fprintf(stderr, "%s: frame %d done — %zu tracked\n", __func__, fi, result.detections.size());
    return result;
}

bool sam3_refine_instance(sam3_tracker& tracker, sam3_state& state,
                          const sam3_model& model, int instance_id,
                          const std::vector<sam3_point>& pos_points,
                          const std::vector<sam3_point>& neg_points) {
    const int D = model.hparams.neck_dim;
    sam3_masklet* tgt = nullptr;
    for (auto& ml : tracker.masklets) if (ml.instance_id == instance_id) { tgt = &ml; break; }
    if (!tgt) for (auto& ml : tracker.pending) if (ml.instance_id == instance_id) { tgt = &ml; break; }
    if (!tgt) { fprintf(stderr, "%s: instance %d not found\n", __func__, instance_id); return false; }
    sam3_pvs_params pvs; pvs.pos_points = pos_points; pvs.neg_points = neg_points; pvs.multimask = false;
    auto r = sam3_segment_pvs(state, model, pvs);
    if (r.detections.empty()) return false;
    tgt->last_score = r.detections[0].score; tgt->last_seen = tracker.frame_index;
    std::vector<float> op(D, 0.0f);
    sam3_store_obj_ptr(tracker, model, instance_id, op.data(), tracker.frame_index);
    fprintf(stderr, "%s: refined instance %d\n", __func__, instance_id);
    return true;
}

int sam3_tracker_frame_index(const sam3_tracker& tracker) { return tracker.frame_index; }

void sam3_tracker_reset(sam3_tracker& tracker) {
    tracker.frame_index = 0; tracker.next_inst_id = 1;
    tracker.masklets.clear(); tracker.pending.clear();
    tracker.mem_banks.clear(); tracker.ptr_banks.clear();
    for (auto* b : tracker.owned_buffers) if (b) ggml_backend_buffer_free(b);
    tracker.owned_buffers.clear();
    if (tracker.ctx) { ggml_free(tracker.ctx); tracker.ctx = nullptr; }
    if (tracker.buffer) { ggml_backend_buffer_free(tracker.buffer); tracker.buffer = nullptr; }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Utility — image I/O
// ═══════════════════════════════════════════════════════════════════════════════

sam3_image sam3_load_image(const std::string& path) {
    sam3_image img;
    int w, h, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, path.c_str());
        return img;
    }
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.assign(data, data + w * h * 3);
    stbi_image_free(data);
    return img;
}

bool sam3_save_mask(const sam3_mask& mask, const std::string& path) {
    if (mask.data.empty()) return false;
    return stbi_write_png(path.c_str(), mask.width, mask.height, 1,
                          mask.data.data(), mask.width) != 0;
}

sam3_image sam3_decode_video_frame(const std::string& video_path, int frame_index) {
    sam3_image img;

    // Use ffmpeg to extract a single frame as raw RGB
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -nostdin -loglevel error -ss %.4f -i \"%s\" "
             "-frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null",
             frame_index / 30.0, video_path.c_str());

    // First, get dimensions
    char info_cmd[1024];
    snprintf(info_cmd, sizeof(info_cmd),
             "ffprobe -v error -select_streams v:0 "
             "-show_entries stream=width,height -of csv=p=0 \"%s\" 2>/dev/null",
             video_path.c_str());
    FILE* fp = popen(info_cmd, "r");
    if (!fp) return img;
    int w = 0, h = 0;
    if (fscanf(fp, "%d,%d", &w, &h) != 2) {
        pclose(fp);
        return img;
    }
    pclose(fp);

    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.resize(w * h * 3);

    fp = popen(cmd, "r");
    if (!fp) {
        img.data.clear();
        return img;
    }
    size_t nread = fread(img.data.data(), 1, img.data.size(), fp);
    pclose(fp);
    if (nread != img.data.size()) {
        img.data.clear();
    }

    return img;
}

sam3_video_info sam3_get_video_info(const std::string& video_path) {
    sam3_video_info info;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams v:0 "
             "-show_entries stream=width,height,r_frame_rate,nb_frames "
             "-of csv=p=0 \"%s\" 2>/dev/null",
             video_path.c_str());
    FILE* fp = popen(cmd, "r");
    if (!fp) return info;

    int w = 0, h = 0, num = 0, den = 1, nf = 0;
    if (fscanf(fp, "%d,%d,%d/%d,%d", &w, &h, &num, &den, &nf) >= 4) {
        info.width = w;
        info.height = h;
        info.fps = (den > 0) ? static_cast<float>(num) / den : 0.0f;
        info.n_frames = nf;
    }
    pclose(fp);
    return info;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tokenizer — standalone test API (does not require model weights)
// ═══════════════════════════════════════════════════════════════════════════════

// Global tokenizer instance for the test API.
static sam3_bpe_tokenizer g_test_tokenizer;
static bool g_test_tokenizer_loaded = false;

bool sam3_test_load_tokenizer(const std::string& dir) {
    if (!sam3_load_bpe_vocab(g_test_tokenizer, dir)) return false;
    g_test_tokenizer_loaded = true;
    return true;
}

std::vector<int32_t> sam3_test_tokenize(const std::string& text) {
    if (!g_test_tokenizer_loaded) return {};
    return sam3_tokenize(g_test_tokenizer, text, 32);
}

static bool sam3_dump_tensor_to_path(struct ggml_tensor * t,
                                     const std::string & tensor_name,
                                     const std::string & output_path) {
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' is null\n", __func__, tensor_name.c_str());
        return false;
    }

    int64_t numel = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) {
            numel *= t->ne[i];
        }
    }

    std::vector<float> data(numel);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16_data(numel);
        ggml_backend_tensor_get(t, f16_data.data(), 0, numel * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(f16_data.data(), data.data(), numel);
    } else if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, numel * sizeof(float));
    } else {
        fprintf(stderr, "%s: unsupported tensor type %d for '%s'\n",
                __func__, (int) t->type, tensor_name.c_str());
        return false;
    }

    {
        std::ofstream f(output_path + ".bin", std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(data.data()), numel * sizeof(float));
    }

    {
        std::ofstream f(output_path + ".shape");
        if (!f) return false;
        int ndims = ggml_n_dims(t);
        for (int i = 0; i < ndims; ++i) {
            if (i > 0) f << ",";
            f << t->ne[i];
        }
        f << "\n";
    }

    fprintf(stderr, "%s: dumped '%s' [", __func__, tensor_name.c_str());
    for (int i = 0; i < ggml_n_dims(t); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%lld", (long long) t->ne[i]);
    }
    fprintf(stderr, "] to %s\n", output_path.c_str());

    return true;
}

static bool sam3_load_ref_f32_data(const std::string & path,
                                   std::vector<float> & data,
                                   int expected_numel = -1) {
    std::ifstream shape_f(path + ".shape");
    if (!shape_f) {
        fprintf(stderr, "%s: missing %s.shape\n", __func__, path.c_str());
        return false;
    }

    int64_t numel = 1;
    std::string line;
    std::getline(shape_f, line);
    size_t pos = 0;
    while (pos < line.size()) {
        size_t end = line.find(',', pos);
        if (end == std::string::npos) {
            end = line.size();
        }
        if (end > pos) {
            numel *= std::stoll(line.substr(pos, end - pos));
        }
        pos = end + 1;
    }

    if (expected_numel >= 0 && numel != expected_numel) {
        fprintf(stderr, "%s: %s expected %d elements, got %lld\n",
                __func__, path.c_str(), expected_numel, (long long) numel);
        return false;
    }

    std::ifstream data_f(path + ".bin", std::ios::binary);
    if (!data_f) {
        fprintf(stderr, "%s: missing %s.bin\n", __func__, path.c_str());
        return false;
    }

    data.resize((size_t) numel);
    data_f.read(reinterpret_cast<char *>(data.data()), numel * sizeof(float));
    return data_f.good() || data_f.eof();
}

static std::vector<float> sam3_reorder_nchw_to_ggml_dwh(const std::vector<float> & src,
                                                        int channels,
                                                        int height,
                                                        int width) {
    std::vector<float> dst((size_t) channels * height * width);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t src_idx = (size_t) c * height * width + (size_t) y * width + x;
                const size_t dst_idx = (size_t) c + (size_t) x * channels + (size_t) y * channels * width;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    return dst;
}

bool sam3_test_dump_text_encoder(const sam3_model & model,
                                 const std::vector<int32_t> & token_ids,
                                 const std::string & output_dir,
                                 int n_threads) {
    const int L = model.hparams.text_ctx_len;
    if ((int) token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }

    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    auto * text_features = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    std::vector<std::string> tensor_names = {
        "causal_mask",
        "text_token_embed",
        "text_after_pos_embed",
        "text_final_ln",
        "text_features_2d",
    };
    for (int i = 0; i < model.hparams.text_layers; ++i) {
        char name[64];

        snprintf(name, sizeof(name), "text_block_%02d_after_ln1", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_qkv", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_attn_out", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_attn_residual", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_ln2", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_fc1", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_gelu", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_out", i);
        tensor_names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_out", i);
        tensor_names.emplace_back(name);
    }

    ggml_set_output(text_features);
    for (const auto & name : tensor_names) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (t && t != inp_tokens) {
            ggml_set_output(t);
        }
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(graph, text_features);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate text encoder graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(inp_tokens, token_ids.data(), 0, L * sizeof(int32_t));

    auto * causal_mask = ggml_get_tensor(ctx0, "causal_mask");
    if (!causal_mask) {
        fprintf(stderr, "%s: causal_mask tensor not found\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    std::vector<ggml_fp16_t> mask_data(L * L);
    sam3_fill_causal_mask(mask_data.data(), L);
    ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto & name : tensor_names) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase5(const sam3_model & model,
                           const sam3_state & state,
                           const std::vector<int32_t> & token_ids,
                           const std::string & output_dir,
                           int n_threads) {
    const auto & hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    const int L = hp.text_ctx_len;
    const int NQ = hp.ddec_num_queries;

    if ((int) token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }
    if (!state.neck_det[0] || !state.neck_det_pe[2]) {
        fprintf(stderr, "%s: encoded detector features are missing\n", __func__);
        return false;
    }

    const size_t buf_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    auto * text_features_2d = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    auto * text_features = ggml_reshape_3d(ctx0, text_features_2d, D, L, 1);
    ggml_set_name(text_features, "text_features");

    auto * img_feats = ggml_reshape_3d(ctx0, state.neck_det[2], D, H * H, 1);
    auto * img_pe = ggml_reshape_3d(ctx0, state.neck_det_pe[2], D, H * H, 1);

    auto * sine_dim_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 64);
    ggml_set_name(sine_dim_t, "sine_dim_t");
    ggml_set_input(sine_dim_t);

    auto * rpb_coords = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, H);
    ggml_set_name(rpb_coords, "rpb_coords");
    ggml_set_input(rpb_coords);

    auto * text_valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_valid_mask, "text_valid_mask");
    ggml_set_input(text_valid_mask);

    auto * text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_attn_bias, "text_attn_bias");
    ggml_set_input(text_attn_bias);

    auto * conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features,
                                               img_pe, text_attn_bias);
    auto * fenc_layer5_out = ggml_cont(ctx0, conditioned);
    ggml_set_name(fenc_layer5_out, "fenc_layer5_out");
    ggml_set_name(conditioned, "fenc_output");

    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features,
                                          sine_dim_t, rpb_coords, text_attn_bias, text_valid_mask);

    struct ggml_tensor * fpn_feats[3] = {
        state.neck_det[0],
        state.neck_det[1],
        state.neck_det[2],
    };

    auto * obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                      ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                      1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    auto * mask_logits = sam3_build_seg_head_graph(ctx0, model, conditioned, fpn_feats,
                                                   obj_queries, text_features, text_attn_bias);
    ggml_set_name(mask_logits, "seg_mask_logits");

    struct named_tensor {
        const char * name;
        struct ggml_tensor * tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"text_features", text_features},
        {"text_valid_mask", text_valid_mask},
        {"fenc_img_input", img_feats},
        {"fenc_pos_embed", img_pe},
        {"fenc_prompt", text_features},
        {"img_pe_72", state.neck_det_pe[2]},
        {"ddec_query_embed", model.ddec.query_embed},
        {"ddec_ref_pts_raw", model.tensors.at("ddec.reference_points.weight")},
        {"ddec_presence_token", model.ddec.presence_token},
        {"ddec_pred_boxes", ddec_out.pred_boxes},
        {"ddec_presence_logit", ddec_out.presence_score},
    };

    std::vector<std::string> named_outputs = {
        "fenc_output",
        "ddec_ref_boxes_init",
        "ddec_query_sine_0",
        "ddec_query_pos_0",
        "ddec_rpb_mask_0",
        "ddec_layer0_after_sa",
        "ddec_layer0_after_text_ca",
        "ddec_layer0_after_img_ca",
        "ddec_layer0_full_out",
        "ddec_layer0_presence",
        "ddec_normed_output",
        "scoring_prompt_mlp_out",
        "scoring_pooled",
        "scoring_proj_pooled",
        "scoring_proj_hs",
        "scoring_class_scores",
        "seg_enc_after_ca",
        "seg_enc_visual",
        "seg_pixel_dec_stage0",
        "seg_pixel_dec_stage1",
        "seg_pixel_decoder_out",
        "seg_instance_embed",
        "seg_mask_embed",
        "seg_mask_logits",
    };

    for (int i = 0; i < hp.fenc_layers; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        named_outputs.emplace_back(name);
    }
    for (int i = 0; i < hp.ddec_layers; ++i) {
        char out_name[64];
        char box_name[64];
        snprintf(out_name, sizeof(out_name), "ddec_layer%d_out", i);
        snprintf(box_name, sizeof(box_name), "ddec_layer%d_refboxes", i);
        named_outputs.emplace_back(out_name);
        named_outputs.emplace_back(box_name);
    }

    ggml_set_output(text_features);
    ggml_set_output(text_valid_mask);
    ggml_set_output(img_feats);
    ggml_set_output(img_pe);
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_set_output(t);
        }
    }
    ggml_set_output(mask_logits);
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);
    ggml_build_forward_expand(graph, text_features);
    ggml_build_forward_expand(graph, text_valid_mask);
    ggml_build_forward_expand(graph, img_feats);
    ggml_build_forward_expand(graph, img_pe);
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_build_forward_expand(graph, t);
        }
    }

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 5 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(inp_tokens, token_ids.data(), 0, L * sizeof(int32_t));

    auto * causal_mask = ggml_get_tensor(ctx0, "causal_mask");
    if (causal_mask) {
        std::vector<ggml_fp16_t> mask_data(L * L);
        sam3_fill_causal_mask(mask_data.data(), L);
        ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));
    }

    {
        std::vector<float> dim_t_data(64);
        for (int i = 0; i < 64; ++i) {
            dim_t_data[i] = 2.0f * (float) M_PI / powf(10000.0f, 2.0f * (float) i / 128.0f);
        }
        ggml_backend_tensor_set(sine_dim_t, dim_t_data.data(), 0, 64 * sizeof(float));
    }

    {
        std::vector<float> coords(H);
        for (int i = 0; i < H; ++i) {
            coords[i] = (float) i / (float) H;
        }
        ggml_backend_tensor_set(rpb_coords, coords.data(), 0, H * sizeof(float));
    }

    auto * qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
    if (qpos_pres) {
        std::vector<float> zeros(D, 0.0f);
        ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
    }

    auto * rpb_pz = ggml_get_tensor(ctx0, "rpb_pres_zeros");
    if (rpb_pz) {
        int n = (int) (rpb_pz->ne[0] * rpb_pz->ne[1] * rpb_pz->ne[2] * rpb_pz->ne[3]);
        std::vector<float> zeros(n, 0.0f);
        ggml_backend_tensor_set(rpb_pz, zeros.data(), 0, n * sizeof(float));
    }

    {
        int n_valid = 0;
        for (int i = 0; i < L; ++i) {
            if (token_ids[i] != 0) {
                ++n_valid;
            }
        }
        if (n_valid == 0) {
            n_valid = 1;
        }

        const float scale = (float) L / (float) n_valid;
        std::vector<float> valid_mask(L);
        std::vector<float> attn_bias(L);
        for (int i = 0; i < L; ++i) {
            const bool is_valid = token_ids[i] != 0;
            valid_mask[i] = is_valid ? scale : 0.0f;
            attn_bias[i] = is_valid ? 0.0f : -1.0e9f;
        }
        ggml_backend_tensor_set(text_valid_mask, valid_mask.data(), 0, L * sizeof(float));
        ggml_backend_tensor_set(text_attn_bias, attn_bias.data(), 0, L * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto & item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model & model,
                                           const std::vector<int32_t> & token_ids,
                                           const std::string & prephase_ref_dir,
                                           const std::string & phase5_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads) {
    const auto & hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    const int L = hp.text_ctx_len;
    const int NQ = hp.ddec_num_queries;
    const int H1 = H * 2;
    const int H0 = H * 4;

    if ((int) token_ids.size() != L) {
        fprintf(stderr, "%s: expected %d token IDs, got %zu\n",
                __func__, L, token_ids.size());
        return false;
    }

    std::vector<float> neck_det_0;
    std::vector<float> neck_det_1;
    std::vector<float> fenc_img_input_data;
    std::vector<float> fenc_pos_embed_data;
    std::vector<float> img_pe_72_data;
    std::vector<float> text_features_data;
    if (!sam3_load_ref_f32_data(prephase_ref_dir + "/neck_det_0", neck_det_0, D * H0 * H0) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_det_1", neck_det_1, D * H1 * H1) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/fenc_img_input", fenc_img_input_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/fenc_pos_embed", fenc_pos_embed_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/img_pe_72", img_pe_72_data, D * H * H) ||
        !sam3_load_ref_f32_data(phase5_ref_dir + "/text_features", text_features_data, D * L)) {
        return false;
    }

    neck_det_0 = sam3_reorder_nchw_to_ggml_dwh(neck_det_0, D, H0, H0);
    neck_det_1 = sam3_reorder_nchw_to_ggml_dwh(neck_det_1, D, H1, H1);

    const size_t buf_size = ggml_tensor_overhead() * 65536 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    auto * text_features = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, L, 1);
    ggml_set_name(text_features, "text_features");
    ggml_set_input(text_features);

    auto * neck0 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H0, H0, 1);
    ggml_set_name(neck0, "ref_neck_det_0");
    ggml_set_input(neck0);
    auto * neck1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H1, H1, 1);
    ggml_set_name(neck1, "ref_neck_det_1");
    ggml_set_input(neck1);
    auto * img_feats = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_feats, "ref_fenc_img_input");
    ggml_set_input(img_feats);
    auto * img_pe = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, H * H, 1);
    ggml_set_name(img_pe, "ref_fenc_pos_embed");
    ggml_set_input(img_pe);
    auto * img_pe_72_cmp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, D * H * H);
    ggml_set_name(img_pe_72_cmp, "ref_img_pe_72_flat");
    ggml_set_input(img_pe_72_cmp);

    auto * sine_dim_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 64);
    ggml_set_name(sine_dim_t, "sine_dim_t");
    ggml_set_input(sine_dim_t);

    auto * rpb_coords = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, H);
    ggml_set_name(rpb_coords, "rpb_coords");
    ggml_set_input(rpb_coords);

    auto * text_valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_valid_mask, "text_valid_mask");
    ggml_set_input(text_valid_mask);

    auto * text_attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L, 1, 1);
    ggml_set_name(text_attn_bias, "text_attn_bias");
    ggml_set_input(text_attn_bias);

    auto * conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features,
                                               img_pe, text_attn_bias);
    auto * fenc_layer5_out = ggml_cont(ctx0, conditioned);
    ggml_set_name(fenc_layer5_out, "fenc_layer5_out");
    ggml_set_name(conditioned, "fenc_output");

    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features,
                                          sine_dim_t, rpb_coords, text_attn_bias, text_valid_mask);

    struct ggml_tensor * fpn_feats[3] = { neck0, neck1, nullptr };

    auto * obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                      ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                      1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    auto * mask_logits = sam3_build_seg_head_graph(ctx0, model, conditioned, fpn_feats,
                                                   obj_queries, text_features, text_attn_bias);
    ggml_set_name(mask_logits, "seg_mask_logits");

    struct named_tensor {
        const char * name;
        struct ggml_tensor * tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"text_features", text_features},
        {"text_valid_mask", text_valid_mask},
        {"fenc_img_input", img_feats},
        {"fenc_pos_embed", img_pe},
        {"fenc_prompt", text_features},
        {"img_pe_72", img_pe_72_cmp},
        {"ddec_query_embed", model.ddec.query_embed},
        {"ddec_ref_pts_raw", model.tensors.at("ddec.reference_points.weight")},
        {"ddec_presence_token", model.ddec.presence_token},
        {"ddec_pred_boxes", ddec_out.pred_boxes},
        {"ddec_presence_logit", ddec_out.presence_score},
    };

    std::vector<std::string> named_outputs = {
        "fenc_output",
        "ddec_ref_boxes_init",
        "ddec_query_sine_0",
        "ddec_query_pos_0",
        "ddec_rpb_mask_0",
        "ddec_layer0_after_sa",
        "ddec_layer0_after_text_ca",
        "ddec_layer0_after_img_ca",
        "ddec_layer0_full_out",
        "ddec_layer0_presence",
        "ddec_normed_output",
        "scoring_prompt_mlp_out",
        "scoring_pooled",
        "scoring_proj_pooled",
        "scoring_proj_hs",
        "scoring_class_scores",
        "seg_enc_after_ca",
        "seg_enc_visual",
        "seg_pixel_dec_stage0",
        "seg_pixel_dec_stage1",
        "seg_pixel_decoder_out",
        "seg_instance_embed",
        "seg_mask_embed",
        "seg_mask_logits",
    };

    for (int i = 0; i < hp.fenc_layers; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        named_outputs.emplace_back(name);
    }
    for (int i = 0; i < hp.ddec_layers; ++i) {
        char out_name[64];
        char box_name[64];
        snprintf(out_name, sizeof(out_name), "ddec_layer%d_out", i);
        snprintf(box_name, sizeof(box_name), "ddec_layer%d_refboxes", i);
        named_outputs.emplace_back(out_name);
        named_outputs.emplace_back(box_name);
    }

    ggml_set_output(text_features);
    ggml_set_output(text_valid_mask);
    ggml_set_output(img_feats);
    ggml_set_output(img_pe);
    ggml_set_output(img_pe_72_cmp);
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_set_output(t);
        }
    }
    ggml_set_output(mask_logits);
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);
    ggml_build_forward_expand(graph, text_features);
    ggml_build_forward_expand(graph, text_valid_mask);
    ggml_build_forward_expand(graph, img_feats);
    ggml_build_forward_expand(graph, img_pe);
    ggml_build_forward_expand(graph, img_pe_72_cmp);
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (t) {
            ggml_build_forward_expand(graph, t);
        }
    }

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 5 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(text_features, text_features_data.data(), 0, text_features_data.size() * sizeof(float));
    ggml_backend_tensor_set(neck0, neck_det_0.data(), 0, neck_det_0.size() * sizeof(float));
    ggml_backend_tensor_set(neck1, neck_det_1.data(), 0, neck_det_1.size() * sizeof(float));
    ggml_backend_tensor_set(img_feats, fenc_img_input_data.data(), 0, fenc_img_input_data.size() * sizeof(float));
    ggml_backend_tensor_set(img_pe, fenc_pos_embed_data.data(), 0, fenc_pos_embed_data.size() * sizeof(float));
    ggml_backend_tensor_set(img_pe_72_cmp, img_pe_72_data.data(), 0, img_pe_72_data.size() * sizeof(float));

    {
        std::vector<float> dim_t_data(64);
        for (int i = 0; i < 64; ++i) {
            dim_t_data[i] = 2.0f * (float) M_PI / powf(10000.0f, 2.0f * (float) i / 128.0f);
        }
        ggml_backend_tensor_set(sine_dim_t, dim_t_data.data(), 0, 64 * sizeof(float));
    }

    {
        std::vector<float> coords(H);
        for (int i = 0; i < H; ++i) {
            coords[i] = (float) i / (float) H;
        }
        ggml_backend_tensor_set(rpb_coords, coords.data(), 0, H * sizeof(float));
    }

    auto * qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
    if (qpos_pres) {
        std::vector<float> zeros(D, 0.0f);
        ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
    }

    auto * rpb_pz = ggml_get_tensor(ctx0, "rpb_pres_zeros");
    if (rpb_pz) {
        int n = (int) (rpb_pz->ne[0] * rpb_pz->ne[1] * rpb_pz->ne[2] * rpb_pz->ne[3]);
        std::vector<float> zeros(n, 0.0f);
        ggml_backend_tensor_set(rpb_pz, zeros.data(), 0, n * sizeof(float));
    }

    {
        int n_valid = 0;
        for (int i = 0; i < L; ++i) {
            if (token_ids[i] != 0) {
                ++n_valid;
            }
        }
        if (n_valid == 0) {
            n_valid = 1;
        }

        const float scale = (float) L / (float) n_valid;
        std::vector<float> valid_mask(L);
        std::vector<float> attn_bias(L);
        for (int i = 0; i < L; ++i) {
            const bool is_valid = token_ids[i] != 0;
            valid_mask[i] = is_valid ? scale : 0.0f;
            attn_bias[i] = is_valid ? 0.0f : -1.0e9f;
        }
        ggml_backend_tensor_set(text_valid_mask, valid_mask.data(), 0, L * sizeof(float));
        ggml_backend_tensor_set(text_attn_bias, attn_bias.data(), 0, L * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto & item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

static bool sam3_test_dump_phase6_impl(const sam3_model & model,
                                       struct ggml_tensor * state_neck0,
                                       struct ggml_tensor * state_neck1,
                                       struct ggml_tensor * state_neck2,
                                       const std::vector<float> * neck0_data,
                                       const std::vector<float> * neck1_data,
                                       const std::vector<float> * neck2_data,
                                       const sam3_pvs_params & params,
                                       const std::string & output_dir,
                                       int n_threads) {
    const auto & hp = model.hparams;
    const int D = hp.sam_embed_dim;
    const int H = hp.n_img_embd();
    const int H1 = H * 2;
    const int H0 = H * 4;

    const size_t buf_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    struct ggml_tensor * neck0 = state_neck0;
    struct ggml_tensor * neck1 = state_neck1;
    struct ggml_tensor * neck2 = state_neck2;
    if (neck0_data && neck1_data && neck2_data) {
        neck0 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H0, H0, 1);
        ggml_set_name(neck0, "sam_dec_feat_s0_input");
        ggml_set_input(neck0);

        neck1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H1, H1, 1);
        ggml_set_name(neck1, "sam_dec_feat_s1_input");
        ggml_set_input(neck1);

        neck2 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, D, H, H, 1);
        ggml_set_name(neck2, "sam_dec_feat_s2_input");
        ggml_set_input(neck2);
    }

    if (!neck0 || !neck1 || !neck2) {
        fprintf(stderr, "%s: missing tracker feature inputs\n", __func__);
        ggml_free(ctx0);
        return false;
    }

    auto pe_out = sam3_build_sam_pe(ctx0, params, D, H);

    auto * no_mem = ggml_reshape_4d(ctx0, model.tensors.at("no_mem_embed"), D, 1, 1, 1);
    auto * image_feats = ggml_add(ctx0, neck2, no_mem);
    ggml_set_name(image_feats, "sam_dec_image_feats");

    auto dec_out = sam3_build_sam_dec_graph(ctx0, model, image_feats,
                                            pe_out.image_pe, pe_out.sparse, pe_out.dense,
                                            neck0, neck1);

    struct named_tensor {
        const char * name;
        struct ggml_tensor * tensor;
    };

    const std::vector<named_tensor> direct_tensors = {
        {"sam_pe_sparse", pe_out.sparse},
        {"sam_pe_dense", pe_out.dense},
        {"sam_pe_image_pe", pe_out.image_pe},
        {"sam_dec_image_feats", image_feats},
        {"sam_dec_sam_token", dec_out.sam_token},
    };

    std::vector<std::string> named_outputs = {
        "sam_dec_tokens_initial",
        "sam_dec_block0_queries",
        "sam_dec_block0_keys",
        "sam_dec_block1_queries",
        "sam_dec_block1_keys",
        "sam_dec_final_queries",
        "sam_dec_feat_s1_proj",
        "sam_dec_feat_s0_proj",
        "sam_dec_upscaled",
        "sam_dec_mask_tokens",
        "sam_dec_masks",
        "sam_dec_iou",
        "sam_dec_obj_score",
    };

    ggml_set_output(dec_out.masks);
    ggml_set_output(dec_out.iou_pred);
    ggml_set_output(dec_out.obj_score);
    ggml_set_output(dec_out.sam_token);
    ggml_set_output(dec_out.mask_tokens);
    for (const auto & item : direct_tensors) {
        ggml_set_output(item.tensor);
    }
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found in graph\n", __func__, name.c_str());
            ggml_free(ctx0);
            return false;
        }
        ggml_set_output(t);
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 32768, false);
    ggml_build_forward_expand(graph, dec_out.masks);
    ggml_build_forward_expand(graph, dec_out.iou_pred);
    ggml_build_forward_expand(graph, dec_out.obj_score);
    ggml_build_forward_expand(graph, dec_out.sam_token);
    ggml_build_forward_expand(graph, dec_out.mask_tokens);
    for (const auto & item : direct_tensors) {
        ggml_build_forward_expand(graph, item.tensor);
    }
    for (const auto & name : named_outputs) {
        ggml_build_forward_expand(graph, ggml_get_tensor(ctx0, name.c_str()));
    }

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "%s: failed to allocate phase 6 graph\n", __func__);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    if (neck0_data && neck1_data && neck2_data) {
        ggml_backend_tensor_set(neck0, neck0_data->data(), 0, neck0_data->size() * sizeof(float));
        ggml_backend_tensor_set(neck1, neck1_data->data(), 0, neck1_data->size() * sizeof(float));
        ggml_backend_tensor_set(neck2, neck2_data->data(), 0, neck2_data->size() * sizeof(float));
    }

    sam3_state pe_cache_state = {};
    sam3_populate_pe_cache(pe_cache_state, model);

    std::vector<float> all_coords;
    std::vector<int> all_labels;
    sam3_collect_pvs_prompt_tokens(params, all_coords, all_labels);
    if ((int) all_labels.size() != pe_out.n_tokens) {
        fprintf(stderr, "%s: prompt token count mismatch (%zu vs %d)\n",
                __func__, all_labels.size(), pe_out.n_tokens);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return false;
    }

    {
        const int num_pos_feats = D / 2;
        std::vector<float> sparse_data((size_t) pe_out.n_tokens * D, 0.0f);
        for (int p = 0; p < pe_out.n_tokens; ++p) {
            float px = all_coords[p * 2 + 0] + 0.5f;
            float py = all_coords[p * 2 + 1] + 0.5f;
            float x_norm = px / (float) hp.img_size;
            float y_norm = py / (float) hp.img_size;
            float pe_vec[256];
            sam3_pe_encode_coord(pe_vec, x_norm, y_norm,
                                 pe_cache_state.pe_gauss_cache.data(), num_pos_feats);

            const int label = all_labels[p];
            if (label == -1) {
                for (int d = 0; d < D; ++d) {
                    sparse_data[(size_t) p * D + d] = pe_cache_state.not_a_point_cache[d];
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    sparse_data[(size_t) p * D + d] = pe_vec[d] + pe_cache_state.point_emb_cache[label][d];
                }
            }
        }

        ggml_backend_tensor_set(pe_out.sparse, sparse_data.data(), 0,
                                sparse_data.size() * sizeof(float));
        ggml_backend_tensor_set(pe_out.image_pe, pe_cache_state.dense_pe_cache.data(), 0,
                                pe_cache_state.dense_pe_cache.size() * sizeof(float));
        ggml_backend_tensor_set(pe_out.dense, pe_cache_state.dense_nomask_cache.data(), 0,
                                pe_cache_state.dense_nomask_cache.size() * sizeof(float));
    }

    sam3_graph_compute(model.backend, graph, n_threads);

    bool ok = true;
    for (const auto & item : direct_tensors) {
        if (!sam3_dump_tensor_to_path(item.tensor, item.name, output_dir + "/" + item.name)) {
            ok = false;
        }
    }
    for (const auto & name : named_outputs) {
        auto * t = ggml_get_tensor(ctx0, name.c_str());
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found after compute\n", __func__, name.c_str());
            ok = false;
            continue;
        }
        if (!sam3_dump_tensor_to_path(t, name, output_dir + "/" + name)) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return ok;
}

bool sam3_test_dump_phase6(const sam3_model & model,
                           const sam3_state & state,
                           const sam3_pvs_params & params,
                           const std::string & output_dir,
                           int n_threads) {
    return sam3_test_dump_phase6_impl(model,
                                      state.neck_trk[0], state.neck_trk[1], state.neck_trk[2],
                                      nullptr, nullptr, nullptr,
                                      params, output_dir, n_threads);
}

bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model & model,
                                           const std::string & prephase_ref_dir,
                                           const sam3_pvs_params & params,
                                           const std::string & output_dir,
                                           int n_threads) {
    const auto & hp = model.hparams;
    const int D = hp.sam_embed_dim;
    const int H = hp.n_img_embd();
    const int H1 = H * 2;
    const int H0 = H * 4;

    std::vector<float> neck_trk_0;
    std::vector<float> neck_trk_1;
    std::vector<float> neck_trk_2;
    if (!sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_0", neck_trk_0, D * H0 * H0) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_1", neck_trk_1, D * H1 * H1) ||
        !sam3_load_ref_f32_data(prephase_ref_dir + "/neck_trk_2", neck_trk_2, D * H * H)) {
        return false;
    }

    neck_trk_0 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_0, D, H0, H0);
    neck_trk_1 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_1, D, H1, H1);
    neck_trk_2 = sam3_reorder_nchw_to_ggml_dwh(neck_trk_2, D, H, H);

    return sam3_test_dump_phase6_impl(model,
                                      nullptr, nullptr, nullptr,
                                      &neck_trk_0, &neck_trk_1, &neck_trk_2,
                                      params, output_dir, n_threads);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Debug: dump state tensors
// ═══════════════════════════════════════════════════════════════════════════════

bool sam3_dump_state_tensor(const sam3_state& state,
                            const std::string& tensor_name,
                            const std::string& output_path) {
    struct ggml_tensor* t = nullptr;

    if (tensor_name == "vit_output") {
        t = state.vit_output;
    } else if (tensor_name == "neck_det_0") {
        t = state.neck_det[0];
    } else if (tensor_name == "neck_det_1") {
        t = state.neck_det[1];
    } else if (tensor_name == "neck_det_2") {
        t = state.neck_det[2];
    } else if (tensor_name == "neck_det_3") {
        t = state.neck_det[3];
    } else if (tensor_name == "neck_trk_0") {
        t = state.neck_trk[0];
    } else if (tensor_name == "neck_trk_1") {
        t = state.neck_trk[1];
    } else if (tensor_name == "neck_trk_2") {
        t = state.neck_trk[2];
    } else if (tensor_name == "neck_trk_3") {
        t = state.neck_trk[3];
    } else if (tensor_name == "neck_det_pe_0") {
        t = state.neck_det_pe[0];
    } else if (tensor_name == "neck_det_pe_1") {
        t = state.neck_det_pe[1];
    } else if (tensor_name == "neck_det_pe_2") {
        t = state.neck_det_pe[2];
    } else if (tensor_name == "neck_det_pe_3") {
        t = state.neck_det_pe[3];
    } else {
        // Search by ggml name in the context
        if (state.ctx) {
            t = ggml_get_tensor(state.ctx, tensor_name.c_str());
        }
        // Also search PE context
        if (!t && state.pe_ctx) {
            t = ggml_get_tensor(state.pe_ctx, tensor_name.c_str());
        }
    }

    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in state\n", __func__, tensor_name.c_str());
        return false;
    }
    return sam3_dump_tensor_to_path(t, tensor_name, output_path);
}
