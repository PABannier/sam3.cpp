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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr uint32_t SAM3_MAGIC   = 0x73616D33; // "sam3"
static constexpr int      SAM3_VERSION = 1;

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal data types — hyperparameters
// ═══════════════════════════════════════════════════════════════════════════════

struct sam3_hparams {
    int32_t img_size            = 1008;
    int32_t patch_size          = 14;
    int32_t vit_embed_dim       = 1024;
    int32_t vit_depth           = 32;
    int32_t vit_num_heads       = 16;
    int32_t vit_mlp_dim         = 4736;   // 1024 * 4.625
    int32_t vit_window_size     = 24;
    int32_t n_global_attn       = 4;
    int32_t global_attn_idx[4]  = {7, 15, 23, 31};

    int32_t text_width          = 1024;
    int32_t text_heads          = 16;
    int32_t text_layers         = 24;
    int32_t text_ctx_len        = 32;
    int32_t text_vocab_size     = 49408;
    int32_t text_out_dim        = 256;

    int32_t neck_dim            = 256;

    int32_t fenc_layers         = 6;
    int32_t fenc_heads          = 8;
    int32_t fenc_ffn_dim        = 2048;

    int32_t ddec_layers         = 6;
    int32_t ddec_heads          = 8;
    int32_t ddec_ffn_dim        = 2048;
    int32_t ddec_num_queries    = 200;

    int32_t geom_layers         = 3;
    int32_t n_presence_tokens   = 1;
    int32_t n_geom_queries      = 4;

    int32_t sam_embed_dim       = 256;
    int32_t sam_dec_depth       = 2;
    int32_t sam_n_multimask     = 3;
    int32_t sam_iou_head_depth  = 3;

    int32_t mem_out_dim         = 64;
    int32_t mem_attn_layers     = 4;
    int32_t num_maskmem         = 7;
    int32_t max_obj_ptrs        = 16;

    int32_t n_amb_experts       = 2;

    // derived helpers
    int32_t n_img_embd()   const { return img_size / patch_size; }           // 72
    int32_t n_img_tokens() const { return n_img_embd() * n_img_embd(); }     // 5184
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }   // 64

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
    struct ggml_tensor * norm1_w    = nullptr;
    struct ggml_tensor * norm1_b    = nullptr;
    struct ggml_tensor * qkv_w      = nullptr;
    struct ggml_tensor * qkv_b      = nullptr;
    struct ggml_tensor * proj_w     = nullptr;
    struct ggml_tensor * proj_b     = nullptr;
    struct ggml_tensor * norm2_w    = nullptr;
    struct ggml_tensor * norm2_b    = nullptr;
    struct ggml_tensor * mlp_fc1_w  = nullptr;
    struct ggml_tensor * mlp_fc1_b  = nullptr;
    struct ggml_tensor * mlp_fc2_w  = nullptr;
    struct ggml_tensor * mlp_fc2_b  = nullptr;
    struct ggml_tensor * freqs_cis  = nullptr;  // [N, 32, 2] RoPE (N=576 window, 5184 global)
};

struct sam3_vit {
    struct ggml_tensor * patch_embed_w = nullptr;  // [embed, 3, patch, patch]
    struct ggml_tensor * pos_embed     = nullptr;  // [1, H, W, embed]
    struct ggml_tensor * ln_pre_w      = nullptr;
    struct ggml_tensor * ln_pre_b      = nullptr;
    std::vector<sam3_vit_block> blocks;
};

// ── Neck (SimpleFPN) ─────────────────────────────────────────────────────────

struct sam3_neck_scale {
    struct ggml_tensor * deconv1_w  = nullptr;
    struct ggml_tensor * deconv1_b  = nullptr;
    struct ggml_tensor * deconv2_w  = nullptr;  // only for 4x scale
    struct ggml_tensor * deconv2_b  = nullptr;
    struct ggml_tensor * conv1x1_w  = nullptr;
    struct ggml_tensor * conv1x1_b  = nullptr;
    struct ggml_tensor * conv3x3_w  = nullptr;
    struct ggml_tensor * conv3x3_b  = nullptr;
};

struct sam3_neck {
    sam3_neck_scale scales[4];
    struct ggml_tensor * norms_w[4] = {};
    struct ggml_tensor * norms_b[4] = {};
};

// ── Text encoder ─────────────────────────────────────────────────────────────

struct sam3_text_block {
    struct ggml_tensor * attn_in_proj_w  = nullptr;
    struct ggml_tensor * attn_in_proj_b  = nullptr;
    struct ggml_tensor * attn_out_proj_w = nullptr;
    struct ggml_tensor * attn_out_proj_b = nullptr;
    struct ggml_tensor * ln1_w = nullptr;
    struct ggml_tensor * ln1_b = nullptr;
    struct ggml_tensor * ln2_w = nullptr;
    struct ggml_tensor * ln2_b = nullptr;
    struct ggml_tensor * mlp_fc1_w = nullptr;
    struct ggml_tensor * mlp_fc1_b = nullptr;
    struct ggml_tensor * mlp_fc2_w = nullptr;
    struct ggml_tensor * mlp_fc2_b = nullptr;
    struct ggml_tensor * ls1 = nullptr;  // LayerScale (may be null)
    struct ggml_tensor * ls2 = nullptr;
};

struct sam3_text_encoder {
    struct ggml_tensor * token_embed_w  = nullptr;  // [vocab, width]
    struct ggml_tensor * pos_embed      = nullptr;  // [ctx_len, width]
    struct ggml_tensor * ln_final_w     = nullptr;
    struct ggml_tensor * ln_final_b     = nullptr;
    struct ggml_tensor * resizer_w      = nullptr;  // [out_dim, width]
    struct ggml_tensor * resizer_b      = nullptr;
    struct ggml_tensor * text_projection = nullptr; // [width, proj_dim]
    std::vector<sam3_text_block> blocks;
};

// ── Fusion encoder ───────────────────────────────────────────────────────────

struct sam3_fenc_layer {
    // self-attention
    struct ggml_tensor * sa_in_proj_w  = nullptr;
    struct ggml_tensor * sa_in_proj_b  = nullptr;
    struct ggml_tensor * sa_out_proj_w = nullptr;
    struct ggml_tensor * sa_out_proj_b = nullptr;
    struct ggml_tensor * norm1_w = nullptr;
    struct ggml_tensor * norm1_b = nullptr;
    // cross-attention to prompt tokens
    struct ggml_tensor * ca_q_w  = nullptr;
    struct ggml_tensor * ca_q_b  = nullptr;
    struct ggml_tensor * ca_kv_w = nullptr;
    struct ggml_tensor * ca_kv_b = nullptr;
    struct ggml_tensor * ca_out_w = nullptr;
    struct ggml_tensor * ca_out_b = nullptr;
    struct ggml_tensor * norm2_w = nullptr;
    struct ggml_tensor * norm2_b = nullptr;
    // FFN
    struct ggml_tensor * ffn_fc1_w = nullptr;
    struct ggml_tensor * ffn_fc1_b = nullptr;
    struct ggml_tensor * ffn_fc2_w = nullptr;
    struct ggml_tensor * ffn_fc2_b = nullptr;
    struct ggml_tensor * norm3_w = nullptr;
    struct ggml_tensor * norm3_b = nullptr;
};

struct sam3_fusion_encoder {
    std::vector<sam3_fenc_layer> layers;
};

// ── DETR decoder ─────────────────────────────────────────────────────────────

struct sam3_ddec_layer {
    // self-attention
    struct ggml_tensor * sa_in_proj_w  = nullptr;
    struct ggml_tensor * sa_in_proj_b  = nullptr;
    struct ggml_tensor * sa_out_proj_w = nullptr;
    struct ggml_tensor * sa_out_proj_b = nullptr;
    struct ggml_tensor * norm1_w = nullptr;
    struct ggml_tensor * norm1_b = nullptr;
    // cross-attention to image
    struct ggml_tensor * ca_q_w   = nullptr;
    struct ggml_tensor * ca_q_b   = nullptr;
    struct ggml_tensor * ca_kv_w  = nullptr;
    struct ggml_tensor * ca_kv_b  = nullptr;
    struct ggml_tensor * ca_out_w = nullptr;
    struct ggml_tensor * ca_out_b = nullptr;
    struct ggml_tensor * norm2_w  = nullptr;
    struct ggml_tensor * norm2_b  = nullptr;
    // cross-attention to text
    struct ggml_tensor * ca_text_q_w   = nullptr;
    struct ggml_tensor * ca_text_q_b   = nullptr;
    struct ggml_tensor * ca_text_kv_w  = nullptr;
    struct ggml_tensor * ca_text_kv_b  = nullptr;
    struct ggml_tensor * ca_text_out_w = nullptr;
    struct ggml_tensor * ca_text_out_b = nullptr;
    struct ggml_tensor * norm3_w = nullptr;
    struct ggml_tensor * norm3_b = nullptr;
    // FFN
    struct ggml_tensor * ffn_fc1_w = nullptr;
    struct ggml_tensor * ffn_fc1_b = nullptr;
    struct ggml_tensor * ffn_fc2_w = nullptr;
    struct ggml_tensor * ffn_fc2_b = nullptr;
    struct ggml_tensor * norm4_w = nullptr;
    struct ggml_tensor * norm4_b = nullptr;
    // box refinement MLP (3 layers)
    struct ggml_tensor * bbox_w[3] = {};
    struct ggml_tensor * bbox_b[3] = {};
};

struct sam3_detr_decoder {
    struct ggml_tensor * query_embed    = nullptr;  // [num_queries, 512]
    struct ggml_tensor * presence_token = nullptr;  // [1, 256]
    // DotProductScoring MLP
    struct ggml_tensor * score_mlp_w[2] = {};
    struct ggml_tensor * score_mlp_b[2] = {};
    struct ggml_tensor * score_ln_w = nullptr;
    struct ggml_tensor * score_ln_b = nullptr;
    // Presence head
    struct ggml_tensor * presence_head_w[2] = {};
    struct ggml_tensor * presence_head_b[2] = {};
    std::vector<sam3_ddec_layer> layers;
};

// ── Geometry / exemplar encoder ──────────────────────────────────────────────

struct sam3_geom_layer {
    struct ggml_tensor * sa_in_proj_w = nullptr;
    struct ggml_tensor * sa_in_proj_b = nullptr;
    struct ggml_tensor * sa_out_proj_w = nullptr;
    struct ggml_tensor * sa_out_proj_b = nullptr;
    struct ggml_tensor * norm1_w = nullptr;
    struct ggml_tensor * norm1_b = nullptr;
    struct ggml_tensor * ca_q_w  = nullptr;
    struct ggml_tensor * ca_q_b  = nullptr;
    struct ggml_tensor * ca_kv_w = nullptr;
    struct ggml_tensor * ca_kv_b = nullptr;
    struct ggml_tensor * ca_out_w = nullptr;
    struct ggml_tensor * ca_out_b = nullptr;
    struct ggml_tensor * norm2_w = nullptr;
    struct ggml_tensor * norm2_b = nullptr;
    struct ggml_tensor * ffn_fc1_w = nullptr;
    struct ggml_tensor * ffn_fc1_b = nullptr;
    struct ggml_tensor * ffn_fc2_w = nullptr;
    struct ggml_tensor * ffn_fc2_b = nullptr;
    struct ggml_tensor * norm3_w = nullptr;
    struct ggml_tensor * norm3_b = nullptr;
};

struct sam3_geom_encoder {
    struct ggml_tensor * point_proj_w = nullptr;
    struct ggml_tensor * point_proj_b = nullptr;
    struct ggml_tensor * box_proj_w   = nullptr;
    struct ggml_tensor * box_proj_b   = nullptr;
    struct ggml_tensor * type_embed   = nullptr;
    struct ggml_tensor * cls_token    = nullptr;
    struct ggml_tensor * post_proj_w  = nullptr;
    struct ggml_tensor * post_proj_b  = nullptr;
    std::vector<sam3_geom_layer> layers;
};

// ── Segmentation head (MaskFormer) ───────────────────────────────────────────

struct sam3_seg_head {
    struct ggml_tensor * up_conv_w[3] = {};
    struct ggml_tensor * up_conv_b[3] = {};
    struct ggml_tensor * up_norm_w[3] = {};
    struct ggml_tensor * up_norm_b[3] = {};
    struct ggml_tensor * ca_prompt_q_w   = nullptr;
    struct ggml_tensor * ca_prompt_q_b   = nullptr;
    struct ggml_tensor * ca_prompt_kv_w  = nullptr;
    struct ggml_tensor * ca_prompt_kv_b  = nullptr;
    struct ggml_tensor * ca_prompt_out_w = nullptr;
    struct ggml_tensor * ca_prompt_out_b = nullptr;
    struct ggml_tensor * mask_embed_w = nullptr;
    struct ggml_tensor * mask_embed_b = nullptr;
};

// ── SAM prompt encoder (tracker path) ────────────────────────────────────────

struct sam3_sam_prompt_enc {
    struct ggml_tensor * pe_gaussian       = nullptr;  // [2, 128]
    struct ggml_tensor * point_embed[4]    = {};       // neg, pos, box_tl, box_br
    struct ggml_tensor * not_a_point_embed = nullptr;  // [256]
    struct ggml_tensor * no_mask_embed     = nullptr;  // [256]
    struct ggml_tensor * mask_ds_conv_w[3] = {};
    struct ggml_tensor * mask_ds_conv_b[3] = {};
    struct ggml_tensor * mask_ds_norm_w[2] = {};
    struct ggml_tensor * mask_ds_norm_b[2] = {};
};

// ── SAM mask decoder (tracker path) ──────────────────────────────────────────

struct sam3_sam_attn {
    struct ggml_tensor * q_w   = nullptr;
    struct ggml_tensor * q_b   = nullptr;
    struct ggml_tensor * k_w   = nullptr;
    struct ggml_tensor * k_b   = nullptr;
    struct ggml_tensor * v_w   = nullptr;
    struct ggml_tensor * v_b   = nullptr;
    struct ggml_tensor * out_w = nullptr;
    struct ggml_tensor * out_b = nullptr;
};

struct sam3_twoway_block {
    sam3_sam_attn self_attn;
    sam3_sam_attn ca_tok2img;
    sam3_sam_attn ca_img2tok;
    struct ggml_tensor * norm1_w = nullptr;
    struct ggml_tensor * norm1_b = nullptr;
    struct ggml_tensor * norm2_w = nullptr;
    struct ggml_tensor * norm2_b = nullptr;
    struct ggml_tensor * norm3_w = nullptr;
    struct ggml_tensor * norm3_b = nullptr;
    struct ggml_tensor * norm4_w = nullptr;
    struct ggml_tensor * norm4_b = nullptr;
    struct ggml_tensor * mlp_fc1_w = nullptr;
    struct ggml_tensor * mlp_fc1_b = nullptr;
    struct ggml_tensor * mlp_fc2_w = nullptr;
    struct ggml_tensor * mlp_fc2_b = nullptr;
};

struct sam3_sam_mask_dec {
    struct ggml_tensor * iou_token       = nullptr;  // [1, 256]
    struct ggml_tensor * mask_tokens     = nullptr;  // [4, 256]
    struct ggml_tensor * obj_score_token = nullptr;  // [1, 256]

    std::vector<sam3_twoway_block> twoway_blocks;     // [2]

    sam3_sam_attn final_attn;
    struct ggml_tensor * final_norm_w = nullptr;
    struct ggml_tensor * final_norm_b = nullptr;

    // upscaling
    struct ggml_tensor * up1_w      = nullptr;
    struct ggml_tensor * up1_b      = nullptr;
    struct ggml_tensor * up1_norm_w = nullptr;
    struct ggml_tensor * up1_norm_b = nullptr;
    struct ggml_tensor * up2_w      = nullptr;
    struct ggml_tensor * up2_b      = nullptr;

    // high-res feature convolutions
    struct ggml_tensor * conv_s0_w = nullptr;
    struct ggml_tensor * conv_s0_b = nullptr;
    struct ggml_tensor * conv_s1_w = nullptr;
    struct ggml_tensor * conv_s1_b = nullptr;

    // hypernetwork MLPs: 4 masks × 3 layers
    struct ggml_tensor * hyper_w[4][3] = {};
    struct ggml_tensor * hyper_b[4][3] = {};

    // IoU prediction head (3 layers)
    struct ggml_tensor * iou_head_w[3] = {};
    struct ggml_tensor * iou_head_b[3] = {};

    // object score head (3 layers)
    struct ggml_tensor * obj_head_w[3] = {};
    struct ggml_tensor * obj_head_b[3] = {};
};

// ── Memory encoder ───────────────────────────────────────────────────────────

struct sam3_mem_enc {
    // mask downsampler (4 conv stages + final 1x1)
    struct ggml_tensor * ds_conv_w[5] = {};
    struct ggml_tensor * ds_conv_b[5] = {};
    struct ggml_tensor * ds_norm_w[4] = {};
    struct ggml_tensor * ds_norm_b[4] = {};
    // pixel feature projection
    struct ggml_tensor * pix_proj_w = nullptr;
    struct ggml_tensor * pix_proj_b = nullptr;
    // fuser (2 CXBlock layers)
    struct ggml_tensor * fuser_dw_w[2]  = {};
    struct ggml_tensor * fuser_dw_b[2]  = {};
    struct ggml_tensor * fuser_norm_w[2] = {};
    struct ggml_tensor * fuser_norm_b[2] = {};
    struct ggml_tensor * fuser_fc1_w[2]  = {};
    struct ggml_tensor * fuser_fc1_b[2]  = {};
    struct ggml_tensor * fuser_fc2_w[2]  = {};
    struct ggml_tensor * fuser_fc2_b[2]  = {};
    struct ggml_tensor * fuser_gamma[2]  = {};
    // output projection
    struct ggml_tensor * out_proj_w = nullptr;
    struct ggml_tensor * out_proj_b = nullptr;
    // temporal pos encodings
    struct ggml_tensor * tpos[7] = {};
};

// ── Memory attention (tracker transformer) ───────────────────────────────────

struct sam3_mem_attn_layer {
    // self-attention (RoPE, 1 head, 256-dim)
    struct ggml_tensor * sa_q_w   = nullptr;
    struct ggml_tensor * sa_q_b   = nullptr;
    struct ggml_tensor * sa_k_w   = nullptr;
    struct ggml_tensor * sa_k_b   = nullptr;
    struct ggml_tensor * sa_v_w   = nullptr;
    struct ggml_tensor * sa_v_b   = nullptr;
    struct ggml_tensor * sa_out_w = nullptr;
    struct ggml_tensor * sa_out_b = nullptr;
    struct ggml_tensor * norm1_w  = nullptr;
    struct ggml_tensor * norm1_b  = nullptr;
    // cross-attention (RoPE, kv_dim=64)
    struct ggml_tensor * ca_q_w   = nullptr;
    struct ggml_tensor * ca_q_b   = nullptr;
    struct ggml_tensor * ca_k_w   = nullptr;  // [256, 64]
    struct ggml_tensor * ca_k_b   = nullptr;
    struct ggml_tensor * ca_v_w   = nullptr;  // [256, 64]
    struct ggml_tensor * ca_v_b   = nullptr;
    struct ggml_tensor * ca_out_w = nullptr;
    struct ggml_tensor * ca_out_b = nullptr;
    struct ggml_tensor * norm2_w  = nullptr;
    struct ggml_tensor * norm2_b  = nullptr;
    // FFN
    struct ggml_tensor * ffn_fc1_w = nullptr;
    struct ggml_tensor * ffn_fc1_b = nullptr;
    struct ggml_tensor * ffn_fc2_w = nullptr;
    struct ggml_tensor * ffn_fc2_b = nullptr;
    struct ggml_tensor * norm3_w   = nullptr;
    struct ggml_tensor * norm3_b   = nullptr;
};

struct sam3_mem_attn {
    std::vector<sam3_mem_attn_layer> layers;
};

// ── BPE tokenizer ────────────────────────────────────────────────────────────

struct sam3_bpe_tokenizer {
    std::unordered_map<std::string, int>              encoder;
    std::unordered_map<int, std::string>              decoder;
    std::vector<std::pair<std::string, std::string>>  merges;
    std::unordered_map<std::string, int>              merge_ranks;  // "a\x1fb" → rank
    std::unordered_map<uint8_t, std::string>          byte_encoder; // byte → unicode UTF-8
    std::unordered_map<std::string, std::string>      cache;
    int sot_token = 49406;
    int eot_token = 49407;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Top-level opaque types (defined here, forward-declared in sam3.h)
// ═══════════════════════════════════════════════════════════════════════════════

struct sam3_model {
    sam3_hparams           hparams;

    sam3_vit               vit;
    sam3_neck              neck_det;
    sam3_neck              neck_trk;
    sam3_text_encoder      text_enc;
    sam3_fusion_encoder    fenc;
    sam3_detr_decoder      ddec;
    sam3_geom_encoder      geom_enc;
    sam3_seg_head          seg_head;

    sam3_sam_prompt_enc    sam_pe;
    sam3_sam_mask_dec      sam_dec;
    sam3_mem_enc           mem_enc;
    sam3_mem_attn          mem_attn;

    // object pointer projection
    struct ggml_tensor * obj_ptr_proj_w[3] = {};
    struct ggml_tensor * obj_ptr_proj_b[3] = {};
    struct ggml_tensor * no_obj_ptr        = nullptr;
    struct ggml_tensor * obj_ptr_tpos_w    = nullptr;
    struct ggml_tensor * obj_ptr_tpos_b    = nullptr;

    // precomputed RoPE frequencies
    struct ggml_tensor * rope_freqs = nullptr;  // [n_img_tokens, head_dim]

    // ggml backend
    struct ggml_context     * ctx     = nullptr;
    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buffer  = nullptr;

    // tensor lookup
    std::map<std::string, struct ggml_tensor *> tensors;

    // tokenizer
    sam3_bpe_tokenizer tokenizer;
};

struct sam3_state {
    // cached backbone outputs
    struct ggml_tensor * vit_output     = nullptr;   // [1, embed, H, W]
    struct ggml_tensor * neck_det[3]    = {};         // FPN levels (det path)
    struct ggml_tensor * neck_trk[3]    = {};         // FPN levels (trk path)
    struct ggml_tensor * neck_det_pe[3] = {};         // sinusoidal PE
    struct ggml_tensor * neck_trk_pe[3] = {};

    int orig_width  = 0;
    int orig_height = 0;

    struct ggml_context     * ctx     = nullptr;
    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buffer  = nullptr;
    struct ggml_gallocr     * galloc  = nullptr;
};

// ── Video tracker state ──────────────────────────────────────────────────────

struct sam3_masklet {
    int   instance_id    = -1;
    int   first_frame    = -1;
    int   last_seen      = -1;
    float last_score     = 0.0f;
    bool  confirmed      = false;
    int   mds_sum        = 0;

    // last predicted mask logits (owned by tracker ctx)
    struct ggml_tensor * mask_logits = nullptr;  // [1, 1, 288, 288]
    struct ggml_tensor * obj_ptr    = nullptr;   // [1, 256]
};

struct sam3_memory_slot {
    struct ggml_tensor * spatial_feats = nullptr;  // [64, 72, 72]
    struct ggml_tensor * spatial_pe    = nullptr;  // [64, 72, 72]
    int frame_index = -1;
    bool is_cond_frame = false;
};

struct sam3_tracker {
    sam3_video_params params;
    int frame_index    = 0;
    int next_inst_id   = 1;

    std::vector<sam3_masklet> masklets;
    std::vector<sam3_masklet> pending;

    std::map<int, std::vector<sam3_memory_slot>>                      mem_banks;
    std::map<int, std::vector<std::pair<int, struct ggml_tensor *>>>  ptr_banks;

    struct ggml_context     * ctx    = nullptr;
    ggml_backend_buffer_t     buffer = nullptr;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal helper declarations
// ═══════════════════════════════════════════════════════════════════════════════

// graph execution
static void sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph, int n_threads);

// ggml building blocks
static struct ggml_tensor * sam3_layer_norm(struct ggml_context * ctx,
                                            struct ggml_tensor * x,
                                            struct ggml_tensor * w,
                                            struct ggml_tensor * b);

static struct ggml_tensor * sam3_layer_norm_2d(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               struct ggml_tensor * w,
                                               struct ggml_tensor * b);

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal helper implementations
// ═══════════════════════════════════════════════════════════════════════════════

static void sam3_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph, int n_threads) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, graph);
}

static struct ggml_tensor * sam3_layer_norm(struct ggml_context * ctx,
                                            struct ggml_tensor * x,
                                            struct ggml_tensor * w,
                                            struct ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-6f);
    x = ggml_mul(ctx, x, w);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static struct ggml_tensor * sam3_layer_norm_2d(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               struct ggml_tensor * w,
                                               struct ggml_tensor * b) {
    // x is [C, H, W, B] in ggml layout — norm over C dimension (dim 0)
    x = ggml_norm(ctx, x, 1e-6f);
    // w, b are [C, 1, 1] — broadcast multiply/add
    x = ggml_mul(ctx, x, w);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
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
static bool sam3_is_letter(const std::string & s, size_t i) {
    uint8_t c = (uint8_t)s[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c >= 0xC0) return true;  // multibyte UTF-8 → treat as letter
    return false;
}

// ── Byte-to-unicode mapping (CLIP / GPT-2 style) ────────────────────────────

// Maps each byte 0-255 to a unique unicode character (as UTF-8 string).
// Printable bytes map to themselves; non-printable bytes map to U+0100..U+0143.
static void sam3_init_byte_encoder(std::unordered_map<uint8_t, std::string> & enc) {
    // Collect printable byte values
    std::vector<int> bs;
    for (int i =  33; i <= 126; ++i) bs.push_back(i);
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
static inline std::string sam3_merge_key(const std::string & a, const std::string & b) {
    std::string k;
    k.reserve(a.size() + 1 + b.size());
    k += a;
    k += '\x1f';
    k += b;
    return k;
}

// ── Minimal JSON parser for vocab.json ───────────────────────────────────────

// Parses a flat { "string": int, ... } JSON object.
static bool sam3_parse_vocab_json(const std::string & path,
                                   std::unordered_map<std::string, int> & encoder) {
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
               (content[pos] == ' '  || content[pos] == '\n' ||
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
                    case '"':  key += '"';  break;
                    case '\\': key += '\\'; break;
                    case '/':  key += '/';  break;
                    case 'n':  key += '\n'; break;
                    case 'r':  key += '\r'; break;
                    case 't':  key += '\t'; break;
                    case 'u': {
                        // Parse 4-hex-digit unicode escape
                        if (pos + 4 >= content.size()) return false;
                        std::string hex = content.substr(pos + 1, 4);
                        int cp = (int)strtol(hex.c_str(), nullptr, 16);
                        key += sam3_codepoint_to_utf8(cp);
                        pos += 4;
                        break;
                    }
                    default: key += content[pos]; break;
                }
            } else {
                key += content[pos];
            }
            pos++;
        }
        if (pos >= content.size()) return false;
        pos++; // skip closing '"'

        // Skip to ':'
        while (pos < content.size() && content[pos] != ':') pos++;
        if (pos >= content.size()) return false;
        pos++;

        // Skip whitespace
        while (pos < content.size() &&
               (content[pos] == ' '  || content[pos] == '\n' ||
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

static bool sam3_load_merges(const std::string & path,
                              std::vector<std::pair<std::string, std::string>> & merges,
                              std::unordered_map<std::string, int> & merge_ranks) {
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

static bool sam3_load_bpe_vocab(sam3_bpe_tokenizer & tok, const std::string & dir) {
    std::string sep(1, '/');
    std::string vocab_path  = dir + sep + "vocab.json";
    std::string merges_path = dir + sep + "merges.txt";

    // Load vocabulary
    if (!sam3_parse_vocab_json(vocab_path, tok.encoder)) {
        fprintf(stderr, "%s: failed to load vocab from '%s'\n", __func__, vocab_path.c_str());
        return false;
    }

    // Build decoder (reverse map)
    tok.decoder.clear();
    for (const auto & kv : tok.encoder) {
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
static std::vector<std::string> sam3_utf8_chars(const std::string & s) {
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
static std::string sam3_bpe_encode(sam3_bpe_tokenizer & tok, const std::string & token) {
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
                best_rank  = it->second;
                best_first  = word[i];
                best_second = word[i + 1];
            }
        }

        if (best_rank == INT_MAX) break;

        // Merge all occurrences of this pair
        std::string merged = best_first + best_second;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
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
static std::vector<std::string> sam3_pretokenize(const std::string & text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        uint8_t c = (uint8_t)text[i];

        // Skip whitespace
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { i++; continue; }

        // Special tokens
        if (i + 15 <= n && text.compare(i, 15, "<|startoftext|>") == 0) {
            tokens.push_back("<|startoftext|>");
            i += 15; continue;
        }
        if (i + 13 <= n && text.compare(i, 13, "<|endoftext|>") == 0) {
            tokens.push_back("<|endoftext|>");
            i += 13; continue;
        }

        // Contractions (must check before letters since ' isn't a letter)
        if (c == '\'') {
            if (i + 2 <= n) {
                char c2 = text[i + 1];
                if (c2 == 's' || c2 == 't' || c2 == 'm' || c2 == 'd') {
                    tokens.push_back(text.substr(i, 2));
                    i += 2; continue;
                }
            }
            if (i + 3 <= n) {
                std::string c3 = text.substr(i + 1, 2);
                if (c3 == "re" || c3 == "ve" || c3 == "ll") {
                    tokens.push_back(text.substr(i, 3));
                    i += 3; continue;
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
            i++; continue;
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
static std::vector<int32_t> sam3_tokenize(sam3_bpe_tokenizer & tok,
                                           const std::string & text,
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
            if (!last_ws) { clean += ' '; last_ws = true; }
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

    for (const auto & word : words) {
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

static bool sam3_load_hparams(std::ifstream & fin, sam3_hparams & hp) {
    auto rd = [&](int32_t & v) { fin.read(reinterpret_cast<char *>(&v), 4); };
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

static void sam3_print_hparams(const sam3_hparams & hp) {
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
static void sam3_register_tensors(sam3_model & model) {
    const auto & hp = model.hparams;
    auto & tensors = model.tensors;
    auto ctx = model.ctx;

    auto T1 = [&](const std::string & name, int64_t d0) -> ggml_tensor * {
        auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T2 = [&](const std::string & name, int64_t d0, int64_t d1) -> ggml_tensor * {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3 = [&](const std::string & name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor * {
        auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4 = [&](const std::string & name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor * {
        auto * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    // Always f32 (for embeddings, biases, norms)
    auto T1f = T1;
    auto T2f = [&](const std::string & name, int64_t d0, int64_t d1) -> ggml_tensor * {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T3f = [&](const std::string & name, int64_t d0, int64_t d1, int64_t d2) -> ggml_tensor * {
        auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d0, d1, d2);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };
    auto T4f = [&](const std::string & name, int64_t d0, int64_t d1, int64_t d2, int64_t d3) -> ggml_tensor * {
        auto * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d0, d1, d2, d3);
        ggml_set_name(t, name.c_str());
        tensors[name] = t;
        return t;
    };

    const int E  = hp.vit_embed_dim;   // 1024
    const int D  = hp.neck_dim;        // 256
    const int TW = hp.text_width;      // 1024
    const int MLP = hp.vit_mlp_dim;    // 4736
    const int FFN = hp.fenc_ffn_dim;   // 2048
    const int NQ  = hp.ddec_num_queries; // 200
    const int MD  = hp.mem_out_dim;    // 64
    const int H   = hp.n_img_embd();   // 72

    // ── ViT backbone ─────────────────────────────────────────────────────
    model.vit.blocks.resize(hp.vit_depth);

    model.vit.patch_embed_w = T4("vit.patch_embed.proj.weight", hp.patch_size, hp.patch_size, 3, E);
    // pos_embed: pretrained [1, 577, 1024] but stored as-is; will be interpolated at runtime if needed
    model.vit.pos_embed     = T3f("vit.pos_embed", E, 577, 1);
    model.vit.ln_pre_w      = T1f("vit.ln_pre.weight", E);
    model.vit.ln_pre_b      = T1f("vit.ln_pre.bias", E);

    for (int i = 0; i < hp.vit_depth; ++i) {
        auto & blk = model.vit.blocks[i];
        auto p = "vit.blocks." + std::to_string(i);
        blk.norm1_w   = T1f(p + ".norm1.weight", E);
        blk.norm1_b   = T1f(p + ".norm1.bias", E);
        blk.qkv_w     = T2(p + ".attn.qkv.weight", E, 3*E);
        blk.qkv_b     = T1f(p + ".attn.qkv.bias", 3*E);
        blk.proj_w    = T2(p + ".attn.proj.weight", E, E);
        blk.proj_b    = T1f(p + ".attn.proj.bias", E);
        blk.norm2_w   = T1f(p + ".norm2.weight", E);
        blk.norm2_b   = T1f(p + ".norm2.bias", E);
        blk.mlp_fc1_w = T2(p + ".mlp.lin1.weight", E, MLP);
        blk.mlp_fc1_b = T1f(p + ".mlp.lin1.bias", MLP);
        blk.mlp_fc2_w = T2(p + ".mlp.lin2.weight", MLP, E);
        blk.mlp_fc2_b = T1f(p + ".mlp.lin2.bias", E);

        // RoPE freqs_cis: [N, 32, 2] where N=5184 for global, 576 for window
        int64_t rope_n = hp.is_global_attn(i) ? hp.n_img_tokens() : (hp.vit_window_size * hp.vit_window_size);
        blk.freqs_cis = T3f(p + ".attn.freqs_cis", 2, 32, rope_n);
    }

    // ── Neck (detector + tracker) ────────────────────────────────────────
    auto register_neck = [&](sam3_neck & neck, const std::string & prefix) {
        // scale 0 (4x): 2 deconvs + conv1x1 + conv3x3
        neck.scales[0].deconv1_w = T4(prefix + "0.dconv_2x2_0.weight", 2, 2, 512, E);
        neck.scales[0].deconv1_b = T1f(prefix + "0.dconv_2x2_0.bias", 512);
        neck.scales[0].deconv2_w = T4(prefix + "0.dconv_2x2_1.weight", 2, 2, D, 512);
        neck.scales[0].deconv2_b = T1f(prefix + "0.dconv_2x2_1.bias", D);
        neck.scales[0].conv1x1_w = T4(prefix + "0.conv_1x1.weight", 1, 1, D, D);
        neck.scales[0].conv1x1_b = T1f(prefix + "0.conv_1x1.bias", D);
        neck.scales[0].conv3x3_w = T4(prefix + "0.conv_3x3.weight", 3, 3, D, D);
        neck.scales[0].conv3x3_b = T1f(prefix + "0.conv_3x3.bias", D);

        // scale 1 (2x): 1 deconv + conv1x1 + conv3x3
        neck.scales[1].deconv1_w = T4(prefix + "1.dconv_2x2.weight", 2, 2, 512, E);
        neck.scales[1].deconv1_b = T1f(prefix + "1.dconv_2x2.bias", 512);
        neck.scales[1].conv1x1_w = T4(prefix + "1.conv_1x1.weight", 1, 1, D, 512);
        neck.scales[1].conv1x1_b = T1f(prefix + "1.conv_1x1.bias", D);
        neck.scales[1].conv3x3_w = T4(prefix + "1.conv_3x3.weight", 3, 3, D, D);
        neck.scales[1].conv3x3_b = T1f(prefix + "1.conv_3x3.bias", D);

        // scale 2 (1x): conv1x1 + conv3x3
        neck.scales[2].conv1x1_w = T4(prefix + "2.conv_1x1.weight", 1, 1, E, D);
        neck.scales[2].conv1x1_b = T1f(prefix + "2.conv_1x1.bias", D);
        neck.scales[2].conv3x3_w = T4(prefix + "2.conv_3x3.weight", 3, 3, D, D);
        neck.scales[2].conv3x3_b = T1f(prefix + "2.conv_3x3.bias", D);

        // scale 3 (0.5x): maxpool(no params) + conv1x1 + conv3x3
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
    model.text_enc.pos_embed     = T2f("text.pos_embed", TW, hp.text_ctx_len);
    model.text_enc.ln_final_w    = T1f("text.ln_final.weight", TW);
    model.text_enc.ln_final_b    = T1f("text.ln_final.bias", TW);
    model.text_enc.resizer_w      = T2("text.resizer.weight", TW, hp.text_out_dim);
    model.text_enc.resizer_b      = T1f("text.resizer.bias", hp.text_out_dim);
    model.text_enc.text_projection = T2f("text.text_projection", TW, 512);  // [1024, 512]

    for (int i = 0; i < hp.text_layers; ++i) {
        auto & blk = model.text_enc.blocks[i];
        auto p = "text.blocks." + std::to_string(i);
        blk.attn_in_proj_w  = T2(p + ".attn.in_proj.weight", TW, 3*TW);
        blk.attn_in_proj_b  = T1f(p + ".attn.in_proj.bias", 3*TW);
        blk.attn_out_proj_w = T2(p + ".attn.out_proj.weight", TW, TW);
        blk.attn_out_proj_b = T1f(p + ".attn.out_proj.bias", TW);
        blk.ln1_w     = T1f(p + ".ln_1.weight", TW);
        blk.ln1_b     = T1f(p + ".ln_1.bias", TW);
        blk.ln2_w     = T1f(p + ".ln_2.weight", TW);
        blk.ln2_b     = T1f(p + ".ln_2.bias", TW);
        blk.mlp_fc1_w = T2(p + ".mlp.fc1.weight", TW, TW*4);
        blk.mlp_fc1_b = T1f(p + ".mlp.fc1.bias", TW*4);
        blk.mlp_fc2_w = T2(p + ".mlp.fc2.weight", TW*4, TW);
        blk.mlp_fc2_b = T1f(p + ".mlp.fc2.bias", TW);
    }

    // ── Fusion encoder ───────────────────────────────────────────────────
    model.fenc.layers.resize(hp.fenc_layers);
    for (int i = 0; i < hp.fenc_layers; ++i) {
        auto & ly = model.fenc.layers[i];
        auto p = "fenc.layers." + std::to_string(i);
        // self-attention
        ly.sa_in_proj_w  = T2(p + ".sa.in_proj_weight", D, 3*D);
        ly.sa_in_proj_b  = T1f(p + ".sa.in_proj_bias", 3*D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        // cross-attention
        ly.ca_q_w  = T2(p + ".ca.in_proj_weight", D, 3*D);
        ly.ca_q_b  = T1f(p + ".ca.in_proj_bias", 3*D);
        ly.ca_kv_w = nullptr; // fused in_proj for MHA
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
    model.ddec.query_embed    = T2f("ddec.query_embed.weight", D, NQ);
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
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }

    // ref_point_head MLP (2 layers: 512→256→256)
    tensors["ddec.ref_point_head.layers.0.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, D);
    tensors["ddec.ref_point_head.layers.0.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    tensors["ddec.ref_point_head.layers.1.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, D);
    tensors["ddec.ref_point_head.layers.1.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    for (auto & kv : std::vector<std::string>{
        "ddec.ref_point_head.layers.0.weight", "ddec.ref_point_head.layers.0.bias",
        "ddec.ref_point_head.layers.1.weight", "ddec.ref_point_head.layers.1.bias"})
        ggml_set_name(tensors[kv], kv.c_str());

    // boxRPB MLPs (x and y, each 2 layers)
    for (const auto & axis : {"x", "y"}) {
        auto bp = std::string("ddec.boxRPB_embed_") + axis;
        tensors[bp + ".layers.0.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2, D);
        tensors[bp + ".layers.0.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        tensors[bp + ".layers.1.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, hp.ddec_heads);
        tensors[bp + ".layers.1.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.ddec_heads);
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
        tensors[bp + ".weight"] = ggml_new_tensor_2d(ctx, (j < 2 ? GGML_TYPE_F16 : GGML_TYPE_F16), D, out);
        ggml_set_name(tensors[bp + ".weight"], (bp + ".weight").c_str());
        tensors[bp + ".bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        ggml_set_name(tensors[bp + ".bias"], (bp + ".bias").c_str());
    }
    tensors["ddec.presence_token_out_norm.weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    tensors["ddec.presence_token_out_norm.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_set_name(tensors["ddec.presence_token_out_norm.weight"], "ddec.presence_token_out_norm.weight");
    ggml_set_name(tensors["ddec.presence_token_out_norm.bias"], "ddec.presence_token_out_norm.bias");

    // DETR decoder layers
    for (int i = 0; i < hp.ddec_layers; ++i) {
        auto & ly = model.ddec.layers[i];
        auto p = "ddec.layers." + std::to_string(i);
        ly.sa_in_proj_w  = T2(p + ".sa.in_proj_weight", D, 3*D);
        ly.sa_in_proj_b  = T1f(p + ".sa.in_proj_bias", 3*D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);

        ly.ca_q_w  = T2(p + ".ca.in_proj_weight", D, 3*D);
        ly.ca_q_b  = T1f(p + ".ca.in_proj_bias", 3*D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w = T1f(p + ".norm2.weight", D);
        ly.norm2_b = T1f(p + ".norm2.bias", D);

        ly.ca_text_q_w  = T2(p + ".ca_text.in_proj_weight", D, 3*D);
        ly.ca_text_q_b  = T1f(p + ".ca_text.in_proj_bias", 3*D);
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
    auto reg = [&](const std::string & n, int64_t d0, int64_t d1, bool is_f32 = false) {
        auto * t = ggml_new_tensor_2d(ctx, is_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, d0, d1);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg1 = [&](const std::string & n, int64_t d0) {
        auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d0);
        ggml_set_name(t, n.c_str());
        tensors[n] = t;
        return t;
    };
    auto reg4 = [&](const std::string & n, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
        auto * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d0, d1, d2, d3);
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
    model.geom_enc.box_proj_w   = T2("geom.boxes_direct_project.weight", 4, D);
    model.geom_enc.box_proj_b   = T1f("geom.boxes_direct_project.bias", D);
    model.geom_enc.type_embed   = T2f("geom.label_embed.weight", D, 2);
    model.geom_enc.cls_token    = T2f("geom.cls_embed.weight", D, 1);
    model.geom_enc.post_proj_w  = T2("geom.final_proj.weight", D, D);
    model.geom_enc.post_proj_b  = T1f("geom.final_proj.bias", D);

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
        auto & ly = model.geom_enc.layers[i];
        auto p = "geom.layers." + std::to_string(i);
        ly.sa_in_proj_w  = T2(p + ".sa.in_proj_weight", D, 3*D);
        ly.sa_in_proj_b  = T1f(p + ".sa.in_proj_bias", 3*D);
        ly.sa_out_proj_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_proj_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w = T1f(p + ".norm1.weight", D);
        ly.norm1_b = T1f(p + ".norm1.bias", D);
        ly.ca_q_w  = T2(p + ".ca.in_proj_weight", D, 3*D);
        ly.ca_q_b  = T1f(p + ".ca.in_proj_bias", 3*D);
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
    model.seg_head.ca_prompt_q_w   = T2("seg.cross_attend_prompt.in_proj_weight", D, 3*D);
    model.seg_head.ca_prompt_q_b   = T1f("seg.cross_attend_prompt.in_proj_bias", 3*D);
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
    model.sam_pe.no_mask_embed     = T2f("sam_pe.no_mask_embed.weight", D, 1);

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
    model.sam_dec.iou_token       = T2f("sam_dec.iou_token.weight", D, 1);
    model.sam_dec.mask_tokens     = T2f("sam_dec.mask_tokens.weight", D, 4);
    model.sam_dec.obj_score_token = T2f("sam_dec.obj_score_token.weight", D, 1);

    model.sam_dec.twoway_blocks.resize(hp.sam_dec_depth);
    for (int i = 0; i < hp.sam_dec_depth; ++i) {
        auto & blk = model.sam_dec.twoway_blocks[i];
        auto p = "sam_dec.twoway." + std::to_string(i);

        auto reg_attn = [&](sam3_sam_attn & a, const std::string & pfx, int in_dim, int out_dim) {
            a.q_w   = T2(pfx + ".q_proj.weight", in_dim, out_dim);
            a.q_b   = T1f(pfx + ".q_proj.bias", out_dim);
            a.k_w   = T2(pfx + ".k_proj.weight", in_dim, out_dim);
            a.k_b   = T1f(pfx + ".k_proj.bias", out_dim);
            a.v_w   = T2(pfx + ".v_proj.weight", in_dim, out_dim);
            a.v_b   = T1f(pfx + ".v_proj.bias", out_dim);
            a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
            a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
        };

        reg_attn(blk.self_attn,   p + ".sa", D, D);
        reg_attn(blk.ca_tok2img,  p + ".cross_attn_token_to_image", D, 128);
        reg_attn(blk.ca_img2tok,  p + ".cross_attn_image_to_token", D, 128);

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
    auto reg_sam_attn = [&](sam3_sam_attn & a, const std::string & pfx, int in_dim, int out_dim) {
        a.q_w   = T2(pfx + ".q_proj.weight", in_dim, out_dim);
        a.q_b   = T1f(pfx + ".q_proj.bias", out_dim);
        a.k_w   = T2(pfx + ".k_proj.weight", in_dim, out_dim);
        a.k_b   = T1f(pfx + ".k_proj.bias", out_dim);
        a.v_w   = T2(pfx + ".v_proj.weight", in_dim, out_dim);
        a.v_b   = T1f(pfx + ".v_proj.bias", out_dim);
        a.out_w = T2(pfx + ".out_proj.weight", out_dim, in_dim);
        a.out_b = T1f(pfx + ".out_proj.bias", in_dim);
    };
    reg_sam_attn(model.sam_dec.final_attn, "sam_dec.final_attn", D, 128);
    model.sam_dec.final_norm_w = T1f("sam_dec.final_norm.weight", D);
    model.sam_dec.final_norm_b = T1f("sam_dec.final_norm.bias", D);

    // upscaling
    model.sam_dec.up1_w      = T4("sam_dec.upscale.0.weight", 2, 2, 64, D);
    model.sam_dec.up1_b      = T1f("sam_dec.upscale.0.bias", 64);
    model.sam_dec.up1_norm_w = T1f("sam_dec.upscale.1.weight", 64);
    model.sam_dec.up1_norm_b = T1f("sam_dec.upscale.1.bias", 64);
    model.sam_dec.up2_w      = T4("sam_dec.upscale.3.weight", 2, 2, 32, 64);
    model.sam_dec.up2_b      = T1f("sam_dec.upscale.3.bias", 32);

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
    int ds_indices[]  = {0, 3, 6, 9, 12};
    int norm_indices[] = {1, 4, 7, 10};
    for (int s = 0; s < 4; ++s) {
        auto si = std::to_string(ds_indices[s]);
        model.mem_enc.ds_conv_w[s] = T4("mem_enc.ds." + si + ".weight", 3, 3, ds_channels[s], ds_channels[s+1]);
        model.mem_enc.ds_conv_b[s] = T1f("mem_enc.ds." + si + ".bias", ds_channels[s+1]);
        auto ni = std::to_string(norm_indices[s]);
        model.mem_enc.ds_norm_w[s] = T1f("mem_enc.ds." + ni + ".weight", ds_channels[s+1]);
        model.mem_enc.ds_norm_b[s] = T1f("mem_enc.ds." + ni + ".bias", ds_channels[s+1]);
    }
    model.mem_enc.ds_conv_w[4] = T4("mem_enc.ds.12.weight", 1, 1, D, D);
    model.mem_enc.ds_conv_b[4] = T1f("mem_enc.ds.12.bias", D);

    model.mem_enc.pix_proj_w = T4("mem_enc.pix_feat_proj.weight", 1, 1, D, D);
    model.mem_enc.pix_proj_b = T1f("mem_enc.pix_feat_proj.bias", D);

    // fuser CXBlocks
    for (int i = 0; i < 2; ++i) {
        auto p = "mem_enc.fuser." + std::to_string(i);
        model.mem_enc.fuser_dw_w[i]   = T4(p + ".dwconv.weight", 7, 7, 1, D);  // groups=256
        model.mem_enc.fuser_dw_b[i]   = T1f(p + ".dwconv.bias", D);
        model.mem_enc.fuser_norm_w[i]  = T1f(p + ".norm.weight", D);
        model.mem_enc.fuser_norm_b[i]  = T1f(p + ".norm.bias", D);
        model.mem_enc.fuser_fc1_w[i]   = T2(p + ".pwconv1.weight", D, 1024);
        model.mem_enc.fuser_fc1_b[i]   = T1f(p + ".pwconv1.bias", 1024);
        model.mem_enc.fuser_fc2_w[i]   = T2(p + ".pwconv2.weight", 1024, D);
        model.mem_enc.fuser_fc2_b[i]   = T1f(p + ".pwconv2.bias", D);
        model.mem_enc.fuser_gamma[i]   = T1f(p + ".gamma", D);
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
        auto & ly = model.mem_attn.layers[i];
        auto p = "mem_attn.layers." + std::to_string(i);
        // self-attention (RoPE, 1 head, 256-dim)
        ly.sa_q_w   = T2(p + ".sa.q_proj.weight", D, D);
        ly.sa_q_b   = T1f(p + ".sa.q_proj.bias", D);
        ly.sa_k_w   = T2(p + ".sa.k_proj.weight", D, D);
        ly.sa_k_b   = T1f(p + ".sa.k_proj.bias", D);
        ly.sa_v_w   = T2(p + ".sa.v_proj.weight", D, D);
        ly.sa_v_b   = T1f(p + ".sa.v_proj.bias", D);
        ly.sa_out_w = T2(p + ".sa.out_proj.weight", D, D);
        ly.sa_out_b = T1f(p + ".sa.out_proj.bias", D);
        ly.norm1_w  = T1f(p + ".norm1.weight", D);
        ly.norm1_b  = T1f(p + ".norm1.bias", D);
        // cross-attention (kv_in_dim=64) — renamed from cross_attn_image → ca
        ly.ca_q_w   = T2(p + ".ca.q_proj.weight", D, D);
        ly.ca_q_b   = T1f(p + ".ca.q_proj.bias", D);
        ly.ca_k_w   = T2(p + ".ca.k_proj.weight", MD, D);
        ly.ca_k_b   = T1f(p + ".ca.k_proj.bias", D);
        ly.ca_v_w   = T2(p + ".ca.v_proj.weight", MD, D);
        ly.ca_v_b   = T1f(p + ".ca.v_proj.bias", D);
        ly.ca_out_w = T2(p + ".ca.out_proj.weight", D, D);
        ly.ca_out_b = T1f(p + ".ca.out_proj.bias", D);
        ly.norm2_w  = T1f(p + ".norm2.weight", D);
        ly.norm2_b  = T1f(p + ".norm2.bias", D);
        // FFN
        ly.ffn_fc1_w = T2(p + ".linear1.weight", D, FFN);
        ly.ffn_fc1_b = T1f(p + ".linear1.bias", FFN);
        ly.ffn_fc2_w = T2(p + ".linear2.weight", FFN, D);
        ly.ffn_fc2_b = T1f(p + ".linear2.bias", D);
        ly.norm3_w   = T1f(p + ".norm3.weight", D);
        ly.norm3_b   = T1f(p + ".norm3.bias", D);
    }

    // ── Object pointer projection ────────────────────────────────────────
    for (int j = 0; j < 3; ++j) {
        auto bp = "obj_ptr_proj.layers." + std::to_string(j);
        model.obj_ptr_proj_w[j] = T2(bp + ".weight", D, D);
        model.obj_ptr_proj_b[j] = T1f(bp + ".bias", D);
    }
    model.no_obj_ptr     = T2f("no_obj_ptr", D, 1);
    model.obj_ptr_tpos_w = T2("obj_ptr_tpos_proj.weight", D, MD);
    model.obj_ptr_tpos_b = T1f("obj_ptr_tpos_proj.bias", MD);

    // standalone tracker params
    // standalone tracker parameters
    T3f("no_mem_embed", D, 1, 1);         // [1, 1, 256]
    T3f("no_mem_pos_enc", D, 1, 1);       // [1, 1, 256]
    T2f("no_obj_embed_spatial", MD, 1);   // [1, 64]
    T4f("trk_mask_ds.weight", 4, 4, 1, 1);  // nn.Conv2d(1,1,4,4): [1,1,4,4]
    T1f("trk_mask_ds.bias", 1);
}

// Load tensors from the binary file into the already-registered ggml tensors
static bool sam3_load_tensors(std::ifstream & fin, sam3_model & model) {
    int n_loaded = 0;
    while (fin.peek() != EOF) {
        int32_t n_dims, name_len, dtype;
        fin.read(reinterpret_cast<char *>(&n_dims),   4);
        fin.read(reinterpret_cast<char *>(&name_len), 4);
        fin.read(reinterpret_cast<char *>(&dtype),    4);
        if (fin.fail()) break;

        // Read shape (reversed in file)
        std::vector<int64_t> shape(n_dims);
        for (int i = 0; i < n_dims; ++i) {
            int32_t d;
            fin.read(reinterpret_cast<char *>(&d), 4);
            shape[i] = d;
        }

        // Read name
        std::string name(name_len, '\0');
        fin.read(&name[0], name_len);

        // Skip to 32-byte alignment
        size_t pos = fin.tellg();
        size_t pad = (32 - pos % 32) % 32;
        if (pad > 0) fin.seekg(pad, std::ios::cur);

        // Look up tensor
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            // Unknown tensor — skip its data
            int64_t n_el = 1;
            for (auto d : shape) n_el *= d;
            size_t bytes = n_el * (dtype == 1 /*f16*/ ? 2 : 4);
            fin.seekg(bytes, std::ios::cur);
            // fprintf(stderr, "  [skip] %s\n", name.c_str());
            continue;
        }

        auto * tensor = it->second;

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
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(buf.data()),
                                  f32_buf.data(), n_el);
            ggml_backend_tensor_set(tensor, f32_buf.data(), 0, n_el * sizeof(float));
        } else if (file_type == GGML_TYPE_F32 && tensor->type == GGML_TYPE_F16) {
            // Convert f32 → f16
            std::vector<ggml_fp16_t> f16_buf(n_el);
            ggml_fp32_to_fp16_row(reinterpret_cast<const float *>(buf.data()),
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

    // Report unloaded tensors for debugging
    int n_unloaded = 0;
    for (const auto & kv : model.tensors) {
        // Check if tensor data is all zeros (likely unloaded)
        // Simple heuristic: check if the tensor name was seen in the file
        // We track this by checking n_loaded vs registered
        (void)kv; // just count
    }
    if (n_loaded < (int)model.tensors.size()) {
        fprintf(stderr, "%s: WARNING: %zu registered tensors were not found in the file\n",
                __func__, model.tensors.size() - n_loaded);
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model loading — public API
// ═══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, params.model_path.c_str());

    std::ifstream fin(params.model_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, params.model_path.c_str());
        return nullptr;
    }

    // ── Read + validate header ───────────────────────────────────────────
    uint32_t magic;
    int32_t version, ftype, n_tensors;
    fin.read(reinterpret_cast<char *>(&magic),     4);
    fin.read(reinterpret_cast<char *>(&version),   4);
    fin.read(reinterpret_cast<char *>(&ftype),     4);
    fin.read(reinterpret_cast<char *>(&n_tensors), 4);

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
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
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
            fprintf(stderr, "%s: WARNING: tokenizer not loaded from '%s' "
                    "(text prompts will not work)\n", __func__, tok_dir.c_str());
        }
    }

    fprintf(stderr, "%s: model loaded successfully\n", __func__);
    return model;
}

void sam3_free_model(sam3_model & model) {
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
void sam3_state_deleter::operator()(sam3_state * p) const {
    if (p) {
        sam3_free_state(*p);
        delete p;
    }
}

void sam3_tracker_deleter::operator()(sam3_tracker * p) const {
    if (p) {
        sam3_tracker_reset(*p);
        delete p;
    }
}

sam3_state_ptr sam3_create_state(const sam3_model & model,
                                const sam3_params & params) {
    sam3_state_ptr state(new sam3_state());

    // TODO: allocate computation context + graph allocator
    //       (Phase 3 of implementation plan)

    fprintf(stderr, "%s: state creation not yet implemented\n", __func__);
    return state;
}

void sam3_free_state(sam3_state & state) {
    if (state.galloc) {
        ggml_gallocr_free(state.galloc);
        state.galloc = nullptr;
    }
    if (state.buffer) {
        ggml_backend_buffer_free(state.buffer);
        state.buffer = nullptr;
    }
    if (state.ctx) {
        ggml_free(state.ctx);
        state.ctx = nullptr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image backbone
// ═══════════════════════════════════════════════════════════════════════════════

bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image) {
    // TODO: preprocess → ViT → neck (det + trk)
    //       cache results in state
    //       (Phase 3 of implementation plan)

    fprintf(stderr, "%s: not yet implemented\n", __func__);
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image segmentation — PCS (text-prompted)
// ═══════════════════════════════════════════════════════════════════════════════

sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params) {
    // TODO: tokenize → text encode → fusion encode → DETR decode
    //       → seg head → NMS → post-process
    //       (Phase 5 of implementation plan)

    fprintf(stderr, "%s: not yet implemented\n", __func__);
    return {};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image segmentation — PVS (visual-prompted)
// ═══════════════════════════════════════════════════════════════════════════════

sam3_result sam3_segment_pvs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pvs_params  & params) {
    // TODO: SAM prompt encode → SAM mask decode → post-process
    //       (Phase 6 of implementation plan)

    fprintf(stderr, "%s: not yet implemented\n", __func__);
    return {};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Video tracking
// ═══════════════════════════════════════════════════════════════════════════════

sam3_tracker_ptr sam3_create_tracker(const sam3_model       & model,
                                    const sam3_video_params & params) {
    sam3_tracker_ptr tracker(new sam3_tracker());
    tracker->params = params;
    return tracker;
}

sam3_result sam3_track_frame(sam3_tracker     & tracker,
                             sam3_state       & state,
                             const sam3_model & model,
                             const sam3_image & frame) {
    // TODO: encode image → detect → propagate → match → update
    //       → memory update → post-process
    //       (Phase 7 of implementation plan)

    fprintf(stderr, "%s: not yet implemented\n", __func__);
    return {};
}

bool sam3_refine_instance(sam3_tracker                   & tracker,
                          sam3_state                     & state,
                          const sam3_model               & model,
                          int                              instance_id,
                          const std::vector<sam3_point>  & pos_points,
                          const std::vector<sam3_point>  & neg_points) {
    // TODO: SAM prompt encode with points → mask decode → update masklet
    //       (Phase 7 of implementation plan)

    fprintf(stderr, "%s: not yet implemented\n", __func__);
    return false;
}

int sam3_tracker_frame_index(const sam3_tracker & tracker) {
    return tracker.frame_index;
}

void sam3_tracker_reset(sam3_tracker & tracker) {
    tracker.frame_index  = 0;
    tracker.next_inst_id = 1;
    tracker.masklets.clear();
    tracker.pending.clear();
    tracker.mem_banks.clear();
    tracker.ptr_banks.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Utility — image I/O
// ═══════════════════════════════════════════════════════════════════════════════

sam3_image sam3_load_image(const std::string & path) {
    sam3_image img;
    int w, h, c;
    uint8_t * data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, path.c_str());
        return img;
    }
    img.width    = w;
    img.height   = h;
    img.channels = 3;
    img.data.assign(data, data + w * h * 3);
    stbi_image_free(data);
    return img;
}

bool sam3_save_mask(const sam3_mask & mask, const std::string & path) {
    if (mask.data.empty()) return false;
    return stbi_write_png(path.c_str(), mask.width, mask.height, 1,
                          mask.data.data(), mask.width) != 0;
}

sam3_image sam3_decode_video_frame(const std::string & video_path, int frame_index) {
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
    FILE * fp = popen(info_cmd, "r");
    if (!fp) return img;
    int w = 0, h = 0;
    if (fscanf(fp, "%d,%d", &w, &h) != 2) { pclose(fp); return img; }
    pclose(fp);

    img.width    = w;
    img.height   = h;
    img.channels = 3;
    img.data.resize(w * h * 3);

    fp = popen(cmd, "r");
    if (!fp) { img.data.clear(); return img; }
    size_t nread = fread(img.data.data(), 1, img.data.size(), fp);
    pclose(fp);
    if (nread != img.data.size()) { img.data.clear(); }

    return img;
}

sam3_video_info sam3_get_video_info(const std::string & video_path) {
    sam3_video_info info;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffprobe -v error -select_streams v:0 "
             "-show_entries stream=width,height,r_frame_rate,nb_frames "
             "-of csv=p=0 \"%s\" 2>/dev/null",
             video_path.c_str());
    FILE * fp = popen(cmd, "r");
    if (!fp) return info;

    int w = 0, h = 0, num = 0, den = 1, nf = 0;
    if (fscanf(fp, "%d,%d,%d/%d,%d", &w, &h, &num, &den, &nf) >= 4) {
        info.width    = w;
        info.height   = h;
        info.fps      = (den > 0) ? static_cast<float>(num) / den : 0.0f;
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
static bool               g_test_tokenizer_loaded = false;

bool sam3_test_load_tokenizer(const std::string & dir) {
    if (!sam3_load_bpe_vocab(g_test_tokenizer, dir)) return false;
    g_test_tokenizer_loaded = true;
    return true;
}

std::vector<int32_t> sam3_test_tokenize(const std::string & text) {
    if (!g_test_tokenizer_loaded) return {};
    return sam3_tokenize(g_test_tokenizer, text, 32);
}
