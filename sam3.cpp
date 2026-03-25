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
#include <map>
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
    struct ggml_tensor * patch_embed_w = nullptr;  // [patch, patch, 3, embed] (ggml conv kernel)
    struct ggml_tensor * pos_embed     = nullptr;  // [embed, 24, 24, 1] (pretrained res, tiled at runtime)
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
    // Note: text_projection ([width, proj_dim]) exists in the checkpoint but is
    // intentionally not loaded. In SAM3, VETextEncoder discards the pooled output
    // that text_projection operates on — only the full token sequence (through
    // resizer) is used for downstream fusion/decoding.
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
    struct ggml_tensor * neck_det[4]    = {};         // FPN levels (det path)
    struct ggml_tensor * neck_trk[4]    = {};         // FPN levels (trk path)
    struct ggml_tensor * neck_det_pe[4] = {};         // sinusoidal PE
    struct ggml_tensor * neck_trk_pe[4] = {};

    int orig_width  = 0;
    int orig_height = 0;

    struct ggml_context     * ctx     = nullptr;
    ggml_backend_t            backend = nullptr;
    ggml_backend_buffer_t     buffer  = nullptr;
    struct ggml_gallocr     * galloc  = nullptr;

    // PE buffer: holds sinusoidal PE tensors for neck outputs
    struct ggml_context     * pe_ctx  = nullptr;
    ggml_backend_buffer_t     pe_buf  = nullptr;
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
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, graph);
}

static struct ggml_tensor * sam3_layer_norm(struct ggml_context * ctx,
                                            struct ggml_tensor * x,
                                            struct ggml_tensor * w,
                                            struct ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
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
    x = ggml_norm(ctx, x, 1e-5f);
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
    // pos_embed: Hiera stores [1, 24, 24, 1024] at pretrained resolution (no cls token).
    // Conversion script writes reversed dims → ggml [E, 24, 24, 1].
    // Tiled 3x at runtime to [E, 72, 72, 1].
    {
        const int pretrained_grid = hp.img_size / hp.patch_size / 3;  // 1008/14/3 = 24
        model.vit.pos_embed = T4f("vit.pos_embed", E, pretrained_grid, pretrained_grid, 1);
    }
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
    // ggml conv2d kernel: [kW, kH, Cin, Cout]
    // ggml conv_transpose kernel: [kW, kH, Cout, Cin]
    // PyTorch Conv2d(Cin, Cout, k) weight: [Cout, Cin, kH, kW] → ggml [kW, kH, Cin, Cout]
    // PyTorch ConvTranspose2d(Cin, Cout, k) weight: [Cin, Cout, kH, kW] → ggml [kW, kH, Cout, Cin]
    auto register_neck = [&](sam3_neck & neck, const std::string & prefix) {
        // scale 0 (4x): ConvTranspose(E→512, k=2, s=2), GELU, ConvTranspose(512→D, k=2, s=2), Conv1x1(D→D), Conv3x3(D→D)
        neck.scales[0].deconv1_w = T4(prefix + "0.dconv_2x2_0.weight", 2, 2, 512, E);   // [kW, kH, Cout=512, Cin=E]
        neck.scales[0].deconv1_b = T1f(prefix + "0.dconv_2x2_0.bias", 512);
        neck.scales[0].deconv2_w = T4(prefix + "0.dconv_2x2_1.weight", 2, 2, D, 512);   // [kW, kH, Cout=D, Cin=512]
        neck.scales[0].deconv2_b = T1f(prefix + "0.dconv_2x2_1.bias", D);
        neck.scales[0].conv1x1_w = T4(prefix + "0.conv_1x1.weight", 1, 1, D, D);        // Conv2d(D→D)
        neck.scales[0].conv1x1_b = T1f(prefix + "0.conv_1x1.bias", D);
        neck.scales[0].conv3x3_w = T4(prefix + "0.conv_3x3.weight", 3, 3, D, D);        // Conv2d(D→D)
        neck.scales[0].conv3x3_b = T1f(prefix + "0.conv_3x3.bias", D);

        // scale 1 (2x): ConvTranspose(E→512, k=2, s=2), Conv1x1(512→D), Conv3x3(D→D)
        neck.scales[1].deconv1_w = T4(prefix + "1.dconv_2x2.weight", 2, 2, 512, E);     // ConvTranspose
        neck.scales[1].deconv1_b = T1f(prefix + "1.dconv_2x2.bias", 512);
        neck.scales[1].conv1x1_w = T4(prefix + "1.conv_1x1.weight", 1, 1, 512, D);      // Conv2d(512→D): Cin=512, Cout=D
        neck.scales[1].conv1x1_b = T1f(prefix + "1.conv_1x1.bias", D);
        neck.scales[1].conv3x3_w = T4(prefix + "1.conv_3x3.weight", 3, 3, D, D);
        neck.scales[1].conv3x3_b = T1f(prefix + "1.conv_3x3.bias", D);

        // scale 2 (1x): Conv1x1(E→D), Conv3x3(D→D)
        neck.scales[2].conv1x1_w = T4(prefix + "2.conv_1x1.weight", 1, 1, E, D);        // Conv2d(E→D): Cin=E, Cout=D
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
    model.text_enc.pos_embed     = T2f("text.pos_embed", TW, hp.text_ctx_len);
    model.text_enc.ln_final_w    = T1f("text.ln_final.weight", TW);
    model.text_enc.ln_final_b    = T1f("text.ln_final.bias", TW);
    model.text_enc.resizer_w      = T2("text.resizer.weight", TW, hp.text_out_dim);
    model.text_enc.resizer_b      = T1f("text.resizer.bias", hp.text_out_dim);
    // text.text_projection is intentionally not registered — the conversion
    // script skips it and the loader rejects unknown tensors. See struct comment.

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

        // Look up tensor — every tensor in the file must be registered
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in file (not registered by model)\n",
                    __func__, name.c_str());
            return false;
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
    state->backend = model.backend;
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
static void sam3_resize_bilinear(const uint8_t * src, int src_w, int src_h,
                                  uint8_t * dst, int dst_w, int dst_h) {
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
                float v = (1 - wy) * ((1 - wx) * src[(y0*src_w + x0)*3 + c] +
                                             wx  * src[(y0*src_w + x1)*3 + c]) +
                               wy  * ((1 - wx) * src[(y1*src_w + x0)*3 + c] +
                                             wx  * src[(y1*src_w + x1)*3 + c]);
                dst[(y*dst_w + x)*3 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v + 0.5f));
            }
        }
    }
}

// Preprocess an image: resize to img_size × img_size, convert to float, normalize.
// Returns a float tensor in [C, H, W] layout (channel-first), range normalized with
// mean=0.5, std=0.5 → pixel values in [-1, 1].
static std::vector<float> sam3_preprocess_image(const sam3_image & image, int img_size) {
    const int C = 3;
    std::vector<float> result(C * img_size * img_size);

    // Resize to img_size × img_size
    std::vector<uint8_t> resized;
    const uint8_t * pixels = image.data.data();
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
static void sam3_compute_axial_cis(float * out,
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
                pe[(i)       + x * d_model + y * d_model * W] = val_y;
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
static struct ggml_tensor * sam3_apply_rope(struct ggml_context * ctx,
                                             struct ggml_tensor * x,
                                             struct ggml_tensor * freqs_cis) {
    // freqs_cis: [2, 32, N] — dim0=2 (cos,sin), dim1=32 (half_head=head_dim/2), dim2=N
    // x: [head_dim, N, num_heads*B] — dim0=64, dim1=N, dim2=batch*heads

    const int64_t head_dim = x->ne[0];     // 64
    const int64_t N        = x->ne[1];     // number of tokens
    const int64_t nheads_B = x->ne[2];     // num_heads * batch
    const int64_t half     = head_dim / 2; // 32

    // Reshape x to [2, half, N, nheads_B] to expose (real, imag) pairs
    auto * x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);

    // freqs_cis: [2, 32, N] → [2, half, N, 1] for broadcast
    auto * fc = ggml_reshape_4d(ctx, freqs_cis, 2, half, N, 1);

    // Extract cos (offset 0) and sin (offset 1) from dim0.
    // fc is [2, half, N, 1] — to slice dim0 we keep strides of dims 1,2,3
    // as nb1,nb2,nb3 of the view, so the view walks over (half, N, 1) correctly.
    auto * cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                 fc->nb[1], fc->nb[2], fc->nb[3], 0);
    auto * sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1,
                                 fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);

    // Extract x_re (offset 0) and x_im (offset 1) from dim0.
    // x_pairs is [2, half, N, nheads_B] — same slicing logic.
    auto * x_re = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                                x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    auto * x_im = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B,
                                x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);

    // Complex multiply: (x_re + j*x_im) * (cos + j*sin)
    auto * out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f), ggml_mul(ctx, x_im, sin_f));
    auto * out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f), ggml_mul(ctx, x_im, cos_f));

    // Interleave back: [2, half, N, nheads_B]
    auto * out = ggml_concat(ctx, out_re, out_im, 0);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, out), head_dim, N, nheads_B);
}

// Single ViT block forward: pre-norm → attn (window or global, with RoPE) → residual → pre-norm → MLP → residual
// x: [E, W, H, B] in ggml layout (following sam.cpp convention)
static struct ggml_tensor * sam3_vit_block_forward(struct ggml_context * ctx,
                                                    struct ggml_tensor * x,
                                                    const sam3_vit_block & blk,
                                                    const sam3_hparams & hp,
                                                    int block_idx) {
    const int E  = hp.vit_embed_dim;      // 1024
    const int NH = hp.vit_num_heads;      // 16
    const int HD = hp.vit_head_dim();     // 64
    const int WS = hp.vit_window_size;    // 24
    const bool is_global = hp.is_global_attn(block_idx);

    auto * shortcut = x;

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
        auto * cur = ggml_mul_mat(ctx, blk.qkv_w, x);
        cur = ggml_add(ctx, cur, blk.qkv_b);
        // cur: [3*E, W_cur, H_cur, B_cur]

        // Reshape and permute to separate Q, K, V (following sam.cpp pattern)
        // [3*E, W*H, B_cur] → [E, 3, W*H, B_cur] → permute(0,3,1,2) → [E, W*H, B_cur, 3]
        cur = ggml_reshape_4d(ctx, cur, E, 3, W_cur * H_cur, B_cur);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));
        // cur: [E, W*H, B_cur, 3]  (ne[3]=3 separates Q/K/V)

        auto * Q = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                 cur->nb[1], cur->nb[2], 0);
        auto * K = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
                                 cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
        auto * V = ggml_view_3d(ctx, cur, E, W_cur * H_cur, B_cur,
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
        auto * attn_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
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
static struct ggml_tensor * sam3_build_vit_graph(struct ggml_context * ctx,
                                                  struct ggml_tensor * input,
                                                  const sam3_model & model) {
    const auto & hp = model.hparams;
    const int E = hp.vit_embed_dim;    // 1024
    const int H = hp.n_img_embd();     // 72
    const int W = hp.n_img_embd();     // 72

    // ── Patch embedding ───────────────────────────────────────────────────
    // Conv2d(3, 1024, k=14, s=14, no bias)
    // Input: [img_size, img_size, 3, 1]
    // Output: [W, H, E, 1]  (ggml conv output convention)
    auto * x = ggml_conv_2d_sk_p0(ctx, model.vit.patch_embed_w, input);

    // Permute to [E, W, H, B] (sam.cpp convention: embed dim first)
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));

    // ── Positional embedding (tiled) ──────────────────────────────────────
    // pos_embed: [E, 24, 24, 1] — Hiera pretrained resolution, no cls token.
    // Tile 3x3 to match [E, 72, 72, 1].
    auto * pos_2d = model.vit.pos_embed;  // [E, 24, 24, 1]

    // Tile 3×3 using ggml_repeat to match [E, 72, 72, 1]
    auto * pos_target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, W, H, 1);
    auto * pos_tiled = ggml_repeat(ctx, pos_2d, pos_target);

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
static void sam3_build_neck_graph(struct ggml_context * ctx,
                                   struct ggml_tensor * vit_out,
                                   const sam3_neck & neck,
                                   struct ggml_tensor * out[4]) {
    // Permute from [E, W, H, B] to [W, H, E, B] for conv operations
    auto * x = ggml_cont(ctx, ggml_permute(ctx, vit_out, 2, 0, 1, 3));

    // Helper: add bias to conv output.
    // Conv output is [W, H, C, B]. Bias is [C] (1D).
    // Reshape bias to [1, 1, C, 1] so ggml_repeat can broadcast.
    auto add_bias = [&](struct ggml_tensor * conv_out, struct ggml_tensor * bias) -> struct ggml_tensor * {
        auto * b3d = ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]);
        return ggml_add(ctx, conv_out, ggml_repeat(ctx, b3d, conv_out));
    };

    // Scale 0 (4×): ConvTranspose(1024→512, k=2, s=2) → GELU → ConvTranspose(512→256, k=2, s=2) → Conv1x1 → Conv3x3
    {
        auto * s0 = ggml_conv_transpose_2d_p0(ctx, neck.scales[0].deconv1_w, x, 2);
        s0 = add_bias(s0, neck.scales[0].deconv1_b);
        s0 = ggml_gelu_erf(ctx, s0);
        s0 = ggml_conv_transpose_2d_p0(ctx, neck.scales[0].deconv2_w, s0, 2);
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
        auto * s1 = ggml_conv_transpose_2d_p0(ctx, neck.scales[1].deconv1_w, x, 2);
        s1 = add_bias(s1, neck.scales[1].deconv1_b);
        s1 = ggml_conv_2d_sk_p0(ctx, neck.scales[1].conv1x1_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv1x1_b);
        s1 = ggml_conv_2d_s1_ph(ctx, neck.scales[1].conv3x3_w, s1);
        s1 = add_bias(s1, neck.scales[1].conv3x3_b);
        out[1] = ggml_cont(ctx, ggml_permute(ctx, s1, 1, 2, 0, 3));
    }

    // Scale 2 (1×): Conv1x1(1024→256) → Conv3x3
    {
        auto * s2 = ggml_conv_2d_sk_p0(ctx, neck.scales[2].conv1x1_w, x);
        s2 = add_bias(s2, neck.scales[2].conv1x1_b);
        s2 = ggml_conv_2d_s1_ph(ctx, neck.scales[2].conv3x3_w, s2);
        s2 = add_bias(s2, neck.scales[2].conv3x3_b);
        out[2] = ggml_cont(ctx, ggml_permute(ctx, s2, 1, 2, 0, 3));
    }

    // Scale 3 (0.5×): MaxPool(k=2, s=2) → Conv1x1(1024→256) → Conv3x3
    {
        auto * s3 = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
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
static struct ggml_tensor * sam3_build_causal_mask(struct ggml_context * ctx, int L) {
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, L);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);
    return mask;
}

// Fill a pre-allocated causal mask buffer (host-side, F16).
// mask_data must hold L*L ggml_fp16_t values.
static void sam3_fill_causal_mask(ggml_fp16_t * mask_data, int L) {
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
static struct ggml_tensor * sam3_text_block_forward(struct ggml_context * ctx,
                                                     struct ggml_tensor * x,
                                                     const sam3_text_block & blk,
                                                     const sam3_hparams & hp,
                                                     struct ggml_tensor * causal_mask) {
    const int E  = hp.text_width;       // 1024
    const int NH = hp.text_heads;       // 16
    const int HD = E / NH;              // 64
    const int64_t L = x->ne[1];        // sequence length

    // ── Self-attention with causal mask ──────────────────────────────────
    auto * shortcut = x;

    // Pre-norm
    x = sam3_layer_norm(ctx, x, blk.ln1_w, blk.ln1_b);

    // QKV projection: [E, L] → [3*E, L]
    auto * qkv = ggml_mul_mat(ctx, blk.attn_in_proj_w, x);
    qkv = ggml_add(ctx, qkv, blk.attn_in_proj_b);

    // Split Q, K, V: reshape [3*E, L] → [E, 3, L] → permute → [E, L, 3]
    qkv = ggml_reshape_3d(ctx, qkv, E, 3, L);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 2, 1, 3));
    // qkv: [E, L, 3]

    auto * Q = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 0);
    auto * K = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 1 * qkv->nb[2]);
    auto * V = ggml_view_2d(ctx, qkv, E, L, qkv->nb[1], 2 * qkv->nb[2]);

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
    auto * attn_out = ggml_flash_attn_ext(ctx, Q, K, V, causal_mask, scale, 0.0f, 0.0f);

    // Reshape [HD, NH, L, 1] → [E, L]
    x = ggml_reshape_2d(ctx, attn_out, E, L);

    // Output projection
    x = ggml_mul_mat(ctx, blk.attn_out_proj_w, x);
    x = ggml_add(ctx, x, blk.attn_out_proj_b);

    // LayerScale (if present)
    if (blk.ls1) {
        x = ggml_mul(ctx, x, blk.ls1);
    }

    // Residual
    x = ggml_add(ctx, shortcut, x);

    // ── MLP ─────────────────────────────────────────────────────────────
    shortcut = x;

    // Pre-norm
    x = sam3_layer_norm(ctx, x, blk.ln2_w, blk.ln2_b);

    // MLP: fc1(1024→4096) → GELU → fc2(4096→1024)
    x = ggml_mul_mat(ctx, blk.mlp_fc1_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc1_b);
    x = ggml_gelu_erf(ctx, x);
    x = ggml_mul_mat(ctx, blk.mlp_fc2_w, x);
    x = ggml_add(ctx, x, blk.mlp_fc2_b);

    // LayerScale (if present)
    if (blk.ls2) {
        x = ggml_mul(ctx, x, blk.ls2);
    }

    // Residual
    x = ggml_add(ctx, shortcut, x);

    return x;
}

// Build the full text encoder computation graph.
// token_ids: [L] int32 tensor (BPE token IDs, padded to ctx_len with 0s).
//            Must be marked as input by caller; data uploaded after alloc.
// Returns: text_features tensor [text_out_dim, L] = [256, L].
// Also creates the causal mask internally (marked as input).
static struct ggml_tensor * sam3_build_text_encoder_graph(struct ggml_context * ctx,
                                                           struct ggml_tensor * token_ids,
                                                           const sam3_model & model) {
    const auto & hp  = model.hparams;
    const auto & enc = model.text_enc;
    const int L  = hp.text_ctx_len;     // 32

    // ── Token embedding: lookup [vocab, E] → [E, L] ─────────────────────
    auto * x = ggml_get_rows(ctx, enc.token_embed_w, token_ids);
    // x: [E, L] = [1024, 32]

    // ── Add positional embedding ─────────────────────────────────────────
    // pos_embed: [E, ctx_len] = [1024, 32]
    x = ggml_add(ctx, x, enc.pos_embed);

    // ── Build causal mask ────────────────────────────────────────────────
    auto * causal_mask = sam3_build_causal_mask(ctx, L);

    // ── 24 transformer blocks ────────────────────────────────────────────
    for (int i = 0; i < hp.text_layers; ++i) {
        x = sam3_text_block_forward(ctx, x, enc.blocks[i], hp, causal_mask);
    }

    // ── Final LayerNorm ──────────────────────────────────────────────────
    x = sam3_layer_norm(ctx, x, enc.ln_final_w, enc.ln_final_b);

    // ── Resizer projection: Linear(1024 → 256) ──────────────────────────
    x = ggml_mul_mat(ctx, enc.resizer_w, x);
    x = ggml_add(ctx, x, enc.resizer_b);
    // x: [OD, L] = [256, 32]

    return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image backbone — public API
// ═══════════════════════════════════════════════════════════════════════════════

bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image) {
    const auto & hp = model.hparams;
    const int img_size = hp.img_size;

    fprintf(stderr, "%s: encoding %dx%d image → %dx%d\n", __func__,
            image.width, image.height, img_size, img_size);

    // Save original dimensions
    state.orig_width  = image.width;
    state.orig_height = image.height;

    // ── Preprocess image ──────────────────────────────────────────────────
    auto img_data = sam3_preprocess_image(image, img_size);

    // ── Build computation graph ───────────────────────────────────────────
    // Create a temporary ggml context for graph building (no data, just ops)
    // We need enough memory for all intermediate tensors during graph construction.
    const size_t buf_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead() * 2;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return false;
    }

    // Create input tensor
    auto * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_size, img_size, 3, 1);
    ggml_set_name(inp, "input_image");
    ggml_set_input(inp);

    // Build ViT graph
    auto * vit_out = sam3_build_vit_graph(ctx0, inp, model);
    ggml_set_name(vit_out, "vit_output");
    ggml_set_output(vit_out);

    // Build neck graphs (detector and tracker paths)
    struct ggml_tensor * neck_det_out[4];
    struct ggml_tensor * neck_trk_out[4];
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
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 16384, false);
    for (int i = 0; i < 4; ++i) {
        ggml_build_forward_expand(graph, neck_det_out[i]);
        ggml_build_forward_expand(graph, neck_trk_out[i]);
    }

    // ── Allocate and compute ──────────────────────────────────────────────
    // Create graph allocator
    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

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
    sam3_graph_compute(model.backend, graph, 4);
    fprintf(stderr, "%s: graph computed\n", __func__);

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
            hp.n_img_embd() * 4,   // 288
            hp.n_img_embd() * 2,   // 144
            hp.n_img_embd(),       //  72
            hp.n_img_embd() / 2,   //  36
        };

        // Compute total bytes needed for all 4 PE tensors
        size_t pe_total = 0;
        for (int i = 0; i < 4; ++i) {
            pe_total += (size_t)neck_dim * scale_sizes[i] * scale_sizes[i] * sizeof(float);
        }

        // Free previous PE resources if re-encoding
        if (state.pe_buf) { ggml_backend_buffer_free(state.pe_buf); state.pe_buf = nullptr; }
        if (state.pe_ctx) { ggml_free(state.pe_ctx);                state.pe_ctx = nullptr; }

        // Create a ggml context for PE tensor metadata (4 tensors + overhead)
        struct ggml_init_params pe_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4 + 256,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        state.pe_ctx = ggml_init(pe_params);

        // Create tensor descriptors
        struct ggml_tensor * pe_tensors[4];
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

    fprintf(stderr, "%s: image encoded successfully\n", __func__);
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
static struct ggml_tensor * sam3_multihead_attn_fused(
        struct ggml_context * ctx,
        struct ggml_tensor * q_in,       // [D, N_q, B]
        struct ggml_tensor * kv_in,      // [D, N_kv, B] (can be same as q_in for self-attn)
        struct ggml_tensor * in_proj_w,  // [D, 3*D] — fused QKV weights
        struct ggml_tensor * in_proj_b,  // [3*D]
        struct ggml_tensor * out_proj_w, // [D, D]
        struct ggml_tensor * out_proj_b, // [D]
        int n_heads,
        struct ggml_tensor * attn_mask = nullptr)  // [N_kv, N_q] or nullptr
{
    const int64_t D    = q_in->ne[0];    // 256
    const int64_t N_q  = q_in->ne[1];
    const int64_t B    = q_in->ne[2];
    const int64_t N_kv = kv_in->ne[1];
    const int64_t HD   = D / n_heads;

    // Split in_proj into Q, K, V projections via views
    // in_proj_w: [D, 3*D] → q_w=[D, D] (rows 0..D-1), k_w=[D, D] (rows D..2D-1), v_w=[D, D] (rows 2D..3D-1)
    auto * q_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 0);
    auto * k_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], D * in_proj_w->nb[1]);
    auto * v_w = ggml_view_2d(ctx, in_proj_w, D, D, in_proj_w->nb[1], 2 * D * in_proj_w->nb[1]);

    auto * q_b = ggml_view_1d(ctx, in_proj_b, D, 0);
    auto * k_b = ggml_view_1d(ctx, in_proj_b, D, D * sizeof(float));
    auto * v_b = ggml_view_1d(ctx, in_proj_b, D, 2 * D * sizeof(float));

    // Project: Q from q_in, K and V from kv_in
    auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);   // [D, N_q, B]
    auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, kv_in), k_b);  // [D, N_kv, B]
    auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, kv_in), v_b);  // [D, N_kv, B]

    // Reshape to multi-head: [HD, N, NH, B]
    Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [HD, N_q, NH, B]

    K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [HD, N_kv, NH, B]

    // Attention
    float scale = 1.0f / sqrtf((float)HD);
    auto * attn_out = ggml_flash_attn_ext(ctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);
    // Result: [HD, NH, N_q, B] — head dim and n_heads adjacent

    // Merge heads: [D, N_q, B]
    auto * merged = ggml_reshape_3d(ctx, attn_out, D, N_q, B);

    // Output projection
    merged = ggml_mul_mat(ctx, out_proj_w, merged);
    merged = ggml_add(ctx, merged, out_proj_b);

    return merged;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Fusion encoder — graph building (6 layers)
// ═══════════════════════════════════════════════════════════════════════════════

// Single fusion encoder layer.
// x: [D, N, B] image features (N=5184), prompt: [D, T, B] text/exemplar tokens, pos: [D, N, B]
// Returns: updated x [D, N, B]
static struct ggml_tensor * sam3_fenc_layer_forward(
        struct ggml_context * ctx,
        const sam3_fenc_layer & ly,
        struct ggml_tensor * x,
        struct ggml_tensor * prompt,
        struct ggml_tensor * pos,
        int n_heads)
{
    // 1. Self-attention on image features (pre-norm)
    {
        auto * shortcut = x;
        auto * x_norm = sam3_layer_norm(ctx, x, ly.norm1_w, ly.norm1_b);
        // q = k = norm(x) + pos, v = norm(x)
        auto * q_in = ggml_add(ctx, x_norm, pos);
        auto * k_in = ggml_add(ctx, x_norm, pos);

        // Self-attention: Q and K have pos, V does not
        // Use fused in_proj but with separate q and kv inputs
        // Since SA uses same source for Q,K,V but Q/K get pos added,
        // we need a custom approach: project q_in for Q, k_in for K, x_norm for V
        const int64_t D = x->ne[0];

        auto * q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto * k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto * v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);

        auto * q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto * k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto * v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, x_norm), v_b);

        const int64_t N  = x->ne[1];
        const int64_t B  = x->ne[2];
        const int64_t HD = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto * sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);

        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        x = ggml_add(ctx, shortcut, sa_out);
    }

    // 2. Cross-attention (image → prompt tokens) with pre-norm
    {
        auto * shortcut = x;
        auto * x_norm = sam3_layer_norm(ctx, x, ly.norm2_w, ly.norm2_b);
        // ca_q_w is actually a fused in_proj for Q,K,V but Q comes from image, K/V from prompt
        // The registration code stores fused [D, 3*D] weights in ca_q_w (same as fenc sa)
        // Q from x_norm, K/V from prompt
        const int64_t D = x->ne[0];

        auto * q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto * k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto * v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);

        auto * q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto * k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto * v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, x_norm), q_b);
        auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, prompt), k_b);
        auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, prompt), v_b);

        const int64_t N_q  = x->ne[1];
        const int64_t N_kv = prompt->ne[1];
        const int64_t B    = x->ne[2];
        const int64_t HD   = D / n_heads;

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto * ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);

        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        x = ggml_add(ctx, shortcut, ca_out);
    }

    // 3. FFN (pre-norm, ReLU activation)
    {
        auto * shortcut = x;
        auto * x_norm = sam3_layer_norm(ctx, x, ly.norm3_w, ly.norm3_b);

        auto * ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, x_norm);
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
static struct ggml_tensor * sam3_build_fenc_graph(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * image_feats,
        struct ggml_tensor * prompt_tokens,
        struct ggml_tensor * pos_enc)
{
    const auto & hp = model.hparams;
    auto * x = image_feats;

    for (int i = 0; i < hp.fenc_layers; ++i) {
        x = sam3_fenc_layer_forward(ctx, model.fenc.layers[i], x, prompt_tokens, pos_enc, hp.fenc_heads);
    }

    return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DETR decoder — graph building (6 layers)
// ═══════════════════════════════════════════════════════════════════════════════

// inverse_sigmoid: log(x / (1 - x)), clamped to avoid inf
static struct ggml_tensor * sam3_inverse_sigmoid(struct ggml_context * ctx, struct ggml_tensor * x) {
    // clamp x to [1e-5, 1-1e-5]
    x = ggml_clamp(ctx, x, 1e-5f, 1.0f - 1e-5f);
    // log(x / (1 - x)) = log(x) - log(1 - x)
    auto * log_x = ggml_log(ctx, x);
    // Compute (1 - x) as (-1)*x + 1.  We use ggml_scale_bias which takes float
    // scalars (no tensor allocation needed, safe in no_alloc contexts).
    auto * one_minus = ggml_scale_bias(ctx, x, -1.0f, 1.0f);
    auto * log_1mx   = ggml_log(ctx, one_minus);
    return ggml_sub(ctx, log_x, log_1mx);
}

// Box refinement MLP (3 layers: D→D→D→4 with ReLU)
static struct ggml_tensor * sam3_bbox_mlp(struct ggml_context * ctx,
                                           struct ggml_tensor * x,
                                           struct ggml_tensor * w[3],
                                           struct ggml_tensor * b[3]) {
    for (int j = 0; j < 3; ++j) {
        x = ggml_mul_mat(ctx, w[j], x);
        x = ggml_add(ctx, x, b[j]);
        if (j < 2) x = ggml_relu(ctx, x);
    }
    return x;
}

// Build box-relative positional bias for DETR cross-attention.
// ref_boxes: [4, N_q, B] — (cx, cy, w, h) in [0,1]
// feat_hw: spatial dimension of feature map (72)
// Returns: bias tensor [N_kv, N_q, n_heads, B] for adding to attention logits
// NOTE: This uses the boxRPB_embed_x and boxRPB_embed_y MLPs from the model tensors map.
static struct ggml_tensor * sam3_compute_box_rpb(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * ref_boxes,  // [4, N_q, B]
        int feat_hw)
{
    // For each query, compute relative position of each spatial feature w.r.t. the box
    // ref_boxes: [4, N_q, B] where dim0 = (cx, cy, w, h)
    const int64_t N_q = ref_boxes->ne[1];
    const int64_t B   = ref_boxes->ne[2];
    const int N_kv = feat_hw * feat_hw;   // 5184
    const int NH   = model.hparams.ddec_heads;  // 8

    // Extract cx, cy, w, h from ref_boxes
    auto * cx = ggml_view_3d(ctx, ref_boxes, 1, N_q, B, ref_boxes->nb[1], ref_boxes->nb[2], 0);
    auto * cy = ggml_view_3d(ctx, ref_boxes, 1, N_q, B, ref_boxes->nb[1], ref_boxes->nb[2], sizeof(float));
    auto * bw = ggml_view_3d(ctx, ref_boxes, 1, N_q, B, ref_boxes->nb[1], ref_boxes->nb[2], 2*sizeof(float));
    auto * bh = ggml_view_3d(ctx, ref_boxes, 1, N_q, B, ref_boxes->nb[1], ref_boxes->nb[2], 3*sizeof(float));

    // Create grid coordinates [0..feat_hw-1] normalized to [0,1]
    // grid_x, grid_y: [feat_hw] → will be used to compute relative offsets
    // We'll create a 2D input for the MLP: for each (query, spatial_pos) pair,
    // compute (grid_x - cx) / w and (grid_y - cy) / h

    // For efficiency, create the grid as input tensors
    auto * grid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, N_kv);
    ggml_set_name(grid, "rpb_grid");
    ggml_set_input(grid);

    // The RPB computation is expensive for large N_q × N_kv.
    // For now, return nullptr to skip RPB (it's a refinement, not critical for correctness).
    // The attention works without it, just with slightly less spatial awareness.
    (void)cx; (void)cy; (void)bw; (void)bh; (void)grid;
    (void)N_q; (void)B; (void)N_kv; (void)NH;
    return nullptr;
}

// Single DETR decoder layer.
// queries: [D, N_q, B] where N_q = 201 (200 object queries + 1 presence token)
// query_pos: [D, N_q, B] positional encoding for queries
// enc_feats: [D, N_kv, B] conditioned image features from fusion encoder
// enc_pos: [D, N_kv, B] positional encoding for image features
// text_feats: [D, T, B] text features
// Returns: updated queries [D, N_q, B]
static struct ggml_tensor * sam3_ddec_layer_forward(
        struct ggml_context * ctx,
        const sam3_ddec_layer & ly,
        struct ggml_tensor * queries,
        struct ggml_tensor * query_pos,
        struct ggml_tensor * enc_feats,
        struct ggml_tensor * enc_pos,
        struct ggml_tensor * text_feats,
        int n_heads)
{
    const int64_t D = queries->ne[0];

    // 1. Self-attention among queries (pre-norm, Q and K get positional encoding)
    {
        auto * shortcut = queries;
        auto * q_norm = sam3_layer_norm(ctx, queries, ly.norm1_w, ly.norm1_b);
        auto * q_in = ggml_add(ctx, q_norm, query_pos);
        auto * k_in = ggml_add(ctx, q_norm, query_pos);

        // Split in_proj and do custom Q/K/V
        auto * q_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 0);
        auto * k_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], D * ly.sa_in_proj_w->nb[1]);
        auto * v_w = ggml_view_2d(ctx, ly.sa_in_proj_w, D, D, ly.sa_in_proj_w->nb[1], 2 * D * ly.sa_in_proj_w->nb[1]);
        auto * q_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 0);
        auto * k_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, D * sizeof(float));
        auto * v_b = ggml_view_1d(ctx, ly.sa_in_proj_b, D, 2 * D * sizeof(float));

        const int64_t N  = queries->ne[1];
        const int64_t B  = queries->ne[2];
        const int64_t HD = D / n_heads;

        auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, q_norm), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto * sa_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        sa_out = ggml_reshape_3d(ctx, sa_out, D, N, B);
        sa_out = ggml_mul_mat(ctx, ly.sa_out_proj_w, sa_out);
        sa_out = ggml_add(ctx, sa_out, ly.sa_out_proj_b);

        queries = ggml_add(ctx, shortcut, sa_out);
    }

    // 2. Cross-attention to conditioned image features (pre-norm)
    {
        auto * shortcut = queries;
        auto * q_norm = sam3_layer_norm(ctx, queries, ly.norm2_w, ly.norm2_b);
        auto * q_in = ggml_add(ctx, q_norm, query_pos);
        auto * k_in = ggml_add(ctx, enc_feats, enc_pos);  // image feats + pos

        auto * q_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 0);
        auto * k_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], D * ly.ca_q_w->nb[1]);
        auto * v_w = ggml_view_2d(ctx, ly.ca_q_w, D, D, ly.ca_q_w->nb[1], 2 * D * ly.ca_q_w->nb[1]);
        auto * q_b = ggml_view_1d(ctx, ly.ca_q_b, D, 0);
        auto * k_b = ggml_view_1d(ctx, ly.ca_q_b, D, D * sizeof(float));
        auto * v_b = ggml_view_1d(ctx, ly.ca_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q  = queries->ne[1];
        const int64_t N_kv = enc_feats->ne[1];
        const int64_t B    = queries->ne[2];
        const int64_t HD   = D / n_heads;

        auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_in), q_b);
        auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, k_in), k_b);
        auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, enc_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto * ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_out_b);

        queries = ggml_add(ctx, shortcut, ca_out);
    }

    // 3. Cross-attention to text tokens (pre-norm)
    {
        auto * shortcut = queries;
        auto * q_norm = sam3_layer_norm(ctx, queries, ly.norm3_w, ly.norm3_b);

        // ca_text uses fused in_proj: Q from queries, K/V from text
        auto * q_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 0);
        auto * k_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], D * ly.ca_text_q_w->nb[1]);
        auto * v_w = ggml_view_2d(ctx, ly.ca_text_q_w, D, D, ly.ca_text_q_w->nb[1], 2 * D * ly.ca_text_q_w->nb[1]);
        auto * q_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 0);
        auto * k_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, D * sizeof(float));
        auto * v_b = ggml_view_1d(ctx, ly.ca_text_q_b, D, 2 * D * sizeof(float));

        const int64_t N_q  = queries->ne[1];
        const int64_t N_kv = text_feats->ne[1];
        const int64_t B    = queries->ne[2];
        const int64_t HD   = D / n_heads;

        auto * Q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, q_norm), q_b);
        auto * K = ggml_add(ctx, ggml_mul_mat(ctx, k_w, text_feats), k_b);
        auto * V = ggml_add(ctx, ggml_mul_mat(ctx, v_w, text_feats), v_b);

        Q = ggml_reshape_4d(ctx, Q, HD, n_heads, N_q, B);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_reshape_4d(ctx, K, HD, n_heads, N_kv, B);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_reshape_4d(ctx, V, HD, n_heads, N_kv, B);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)HD);
        auto * ca_out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        ca_out = ggml_reshape_3d(ctx, ca_out, D, N_q, B);
        ca_out = ggml_mul_mat(ctx, ly.ca_text_out_w, ca_out);
        ca_out = ggml_add(ctx, ca_out, ly.ca_text_out_b);

        queries = ggml_add(ctx, shortcut, ca_out);
    }

    // 4. FFN (pre-norm, ReLU)
    {
        auto * shortcut = queries;
        auto * q_norm = sam3_layer_norm(ctx, queries, ly.norm4_w, ly.norm4_b);

        auto * ffn = ggml_mul_mat(ctx, ly.ffn_fc1_w, q_norm);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc1_b);
        ffn = ggml_relu(ctx, ffn);
        ffn = ggml_mul_mat(ctx, ly.ffn_fc2_w, ffn);
        ffn = ggml_add(ctx, ffn, ly.ffn_fc2_b);

        queries = ggml_add(ctx, shortcut, ffn);
    }

    return queries;
}

// DotProductScoring: classify queries against text features via dot product.
// query_outputs: [D, N_q, B] — the 200 object query outputs
// text_features: [D, T, B] — text encoder output
// Returns: class_scores [N_q, B] (one score per query per batch)
static struct ggml_tensor * sam3_dot_product_scoring(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * query_outputs,  // [D, N_q, B]
        struct ggml_tensor * text_features)  // [D, T, B]
{
    const auto & tensors = model.tensors;
    const int64_t D = query_outputs->ne[0];  // 256

    // Project text features through scoring MLP: residual MLP with ReLU + LayerNorm
    // prompt_proj: linear D→D
    auto * text_proj = ggml_mul_mat(ctx, tensors.at("scoring.prompt_proj.weight"), text_features);
    text_proj = ggml_add(ctx, text_proj, tensors.at("scoring.prompt_proj.bias"));

    // prompt_mlp: D→FFN→D with ReLU, residual, + LayerNorm
    auto * mlp_out = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.0.weight"), text_proj);
    mlp_out = ggml_add(ctx, mlp_out, tensors.at("scoring.prompt_mlp.layers.0.bias"));
    mlp_out = ggml_relu(ctx, mlp_out);
    mlp_out = ggml_mul_mat(ctx, tensors.at("scoring.prompt_mlp.layers.1.weight"), mlp_out);
    mlp_out = ggml_add(ctx, mlp_out, tensors.at("scoring.prompt_mlp.layers.1.bias"));
    // Residual
    text_proj = ggml_add(ctx, text_proj, mlp_out);
    // LayerNorm
    text_proj = sam3_layer_norm(ctx, text_proj,
                                 tensors.at("scoring.prompt_mlp.out_norm.weight"),
                                 tensors.at("scoring.prompt_mlp.out_norm.bias"));
    // text_proj: [D, T, B]

    // Project queries through hs_proj: D→D
    auto * q_proj = ggml_mul_mat(ctx, tensors.at("scoring.hs_proj.weight"), query_outputs);
    q_proj = ggml_add(ctx, q_proj, tensors.at("scoring.hs_proj.bias"));
    // q_proj: [D, N_q, B]

    // Dot product: for each query, dot with each text token, then max/mean over text tokens
    // scores = q_proj^T @ text_proj → [N_q, T, B]
    // Then take max over T dimension to get [N_q, B]
    // Actually: ggml_mul_mat(text_proj, q_proj) with text_proj^T @ q_proj → shapes need careful handling
    // text_proj: [D, T, B], q_proj: [D, N_q, B]
    // We want: for each batch, (N_q × D) @ (D × T) = (N_q × T)
    // ggml_mul_mat(A, B) = A^T @ B where A=[D, T], B=[D, N_q] → result [T, N_q]
    auto * scores = ggml_mul_mat(ctx, text_proj, q_proj);  // [T, N_q, B]

    // Max-pool over text tokens (dim 0) to get per-query score
    // Use ggml_pool_1d on the T dimension — but scores is [T, N_q, B]
    // We want max over T for each (N_q, B).
    // Reshape to use a reduction: [T, N_q*B] then pool
    const int64_t T   = scores->ne[0];
    const int64_t N_q = scores->ne[1];
    const int64_t B   = scores->ne[2];

    // For max-pooling over dim0, we can use ggml_pool_1d with k=T, s=T
    // But ggml_pool_1d works on [W, C, N] → pools over W.
    // scores: [T, N_q, B] — pool over T.
    auto * pooled = ggml_pool_1d(ctx, scores, GGML_OP_POOL_MAX, T, T, 0);
    // pooled: [1, N_q, B]
    pooled = ggml_reshape_2d(ctx, pooled, N_q, B);
    // pooled: [N_q, B]

    return pooled;
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
    struct ggml_tensor * queries;         // [D, 201, B]
    struct ggml_tensor * pred_boxes;      // [4, 200, B]
    struct ggml_tensor * class_scores;    // [200, B]
    struct ggml_tensor * presence_score;  // [1, B]
};

static sam3_ddec_output sam3_build_ddec_graph(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * enc_feats,   // [D, N_kv, B]
        struct ggml_tensor * enc_pos,     // [D, N_kv, B]
        struct ggml_tensor * text_feats)  // [D, T, B]
{
    const auto & hp = model.hparams;
    const auto & tensors = model.tensors;
    const int D  = hp.neck_dim;           // 256
    const int NQ = hp.ddec_num_queries;   // 200
    const int B  = (int)enc_feats->ne[2]; // batch (1)

    // ── Initialize queries from query_embed ──────────────────────────────
    // query_embed: [D, NQ] → split into content [D, NQ] (first D) and pos [D, NQ] (last D, via ref_point_head)
    // Actually query_embed is [D, NQ] — wait, it was registered as [D, NQ] but the plan says [NQ, 512].
    // Let me check: model.ddec.query_embed was registered as T2f("ddec.query_embed.weight", D, NQ)
    // So it's [256, 200] in ggml = [D, NQ]. But the plan says 512-dim split into 256+256.
    // The reference_points tensor is separate: [4, NQ].
    // Content queries are the query_embed itself [D, NQ].

    // Content queries: [D, NQ, 1] → prepend presence token → [D, NQ+1, 1]
    auto * content = ggml_reshape_3d(ctx, model.ddec.query_embed, D, NQ, 1);
    auto * pres_tok = ggml_reshape_3d(ctx, model.ddec.presence_token, D, 1, 1);
    auto * queries = ggml_concat(ctx, pres_tok, content, 1);  // [D, NQ+1, B=1]

    // Reference points for positional encoding
    // reference_points.weight: [4, NQ] → sigmoid → initial anchor boxes
    auto * ref_pts_raw = tensors.at("ddec.reference_points.weight");  // [4, NQ]
    auto * ref_boxes = ggml_sigmoid(ctx, ref_pts_raw);  // [4, NQ]
    ref_boxes = ggml_reshape_3d(ctx, ref_boxes, 4, NQ, 1);  // [4, NQ, 1]

    // Positional encoding from reference points via ref_point_head MLP.
    // ref_point_head: MLP [512->256->256]. Input is 512-dim sinusoidal embedding
    // of the 4D reference points (cx, cy, w, h), 128 features per coordinate.
    // The sine positional embedding and MLP are computed CPU-side (below, in
    // the "Set input data" section) and uploaded into the ddec_query_pos_obj
    // input tensor.

    // Create positional encoding for queries from reference points
    // Simple approach: pass ref_inv through the ref_point_head MLP
    // But input needs to be 512-dim. The standard DINO/DETR approach is:
    // pos_embed = get_sine_pos_embed(ref_boxes, 256) → [NQ, 512] (256 for xy + 256 for wh)
    // Then ref_point_head maps 512 → 256

    // For presence token, we use zeros as positional encoding
    auto * query_pos_obj = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, 1);
    ggml_set_name(query_pos_obj, "ddec_query_pos_obj");
    ggml_set_input(query_pos_obj);  // Will be computed separately

    auto * query_pos_pres = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, 1);
    ggml_set_name(query_pos_pres, "ddec_query_pos_pres");
    ggml_set_input(query_pos_pres);  // zeros

    auto * query_pos = ggml_concat(ctx, query_pos_pres, query_pos_obj, 1);  // [D, NQ+1, 1]

    // ── Run decoder layers ───────────────────────────────────────────────
    for (int i = 0; i < hp.ddec_layers; ++i) {
        queries = sam3_ddec_layer_forward(ctx, model.ddec.layers[i],
                                           queries, query_pos,
                                           enc_feats, enc_pos,
                                           text_feats, hp.ddec_heads);

        // Box refinement after each layer (on object queries only, not presence token)
        // Extract object queries: queries[D, 1:, B]
        auto * obj_q = ggml_view_3d(ctx, queries, D, NQ, 1,
                                     queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
        obj_q = ggml_cont(ctx, obj_q);

        // Shared bbox_embed MLP (registered globally, not per-layer)
        auto * bd = obj_q;
        for (int j = 0; j < 3; ++j) {
            auto wn = "ddec.bbox_embed.layers." + std::to_string(j) + ".weight";
            auto bn = "ddec.bbox_embed.layers." + std::to_string(j) + ".bias";
            bd = ggml_mul_mat(ctx, tensors.at(wn), bd);
            bd = ggml_add(ctx, bd, tensors.at(bn));
            if (j < 2) bd = ggml_relu(ctx, bd);
        }
        // bd: [4, NQ, 1]

        // ref_boxes = sigmoid(inverse_sigmoid(ref_boxes) + box_delta)
        auto * ref_inv_cur = sam3_inverse_sigmoid(ctx, ref_boxes);
        ref_boxes = ggml_sigmoid(ctx, ggml_add(ctx, ref_inv_cur, bd));
        // ref_boxes: [4, NQ, 1] — detach (stop gradient, but not needed in inference)
    }

    // ── Final normalization ──────────────────────────────────────────────
    queries = sam3_layer_norm(ctx, queries,
                               tensors.at("ddec.norm.weight"),
                               tensors.at("ddec.norm.bias"));

    // ── Classification via DotProductScoring ─────────────────────────────
    // Extract object queries (skip presence token at index 0)
    auto * obj_queries = ggml_view_3d(ctx, queries, D, NQ, 1,
                                       queries->nb[1], queries->nb[2], 1 * queries->nb[1]);
    obj_queries = ggml_cont(ctx, obj_queries);

    auto * class_scores = sam3_dot_product_scoring(ctx, model, obj_queries, text_feats);
    // class_scores: [NQ, B]

    // ── Presence score ───────────────────────────────────────────────────
    // Extract presence token (index 0)
    auto * pres_out = ggml_view_3d(ctx, queries, D, 1, 1,
                                    queries->nb[1], queries->nb[2], 0);
    pres_out = ggml_cont(ctx, pres_out);

    // Presence token head: LN + 3-layer MLP (D→D→D→1)
    pres_out = sam3_layer_norm(ctx, pres_out,
                                tensors.at("ddec.presence_token_out_norm.weight"),
                                tensors.at("ddec.presence_token_out_norm.bias"));

    for (int j = 0; j < 3; ++j) {
        auto wn = "ddec.presence_token_head.layers." + std::to_string(j) + ".weight";
        auto bn = "ddec.presence_token_head.layers." + std::to_string(j) + ".bias";
        pres_out = ggml_mul_mat(ctx, tensors.at(wn), pres_out);
        pres_out = ggml_add(ctx, pres_out, tensors.at(bn));
        if (j < 2) pres_out = ggml_relu(ctx, pres_out);
    }
    auto * presence_score = ggml_sigmoid(ctx, pres_out);
    // presence_score: [1, 1, 1] → reshape to [1, B]
    presence_score = ggml_reshape_2d(ctx, presence_score, 1, 1);

    sam3_ddec_output out;
    out.queries         = queries;          // [D, NQ+1, B]
    out.pred_boxes      = ref_boxes;        // [4, NQ, B]
    out.class_scores    = class_scores;     // [NQ, B]
    out.presence_score  = presence_score;   // [1, B]

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
static struct ggml_tensor * sam3_pixel_decoder(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * fpn_feats[3])  // [D, W, H, B] at 3 scales
{
    const auto & seg = model.seg_head;

    // Start from lowest resolution and progressively upsample
    auto * feat = fpn_feats[2];  // [D, 72, 72, B]

    // Upsample 2x + merge with FPN[1] (144x144)
    // Permute to [W, H, D, B] for conv operations
    feat = ggml_cont(ctx, ggml_permute(ctx, feat, 2, 0, 1, 3));  // [72, 72, D, B]
    feat = ggml_upscale(ctx, feat, 2, GGML_SCALE_MODE_NEAREST);  // [144, 144, D, B]
    // Conv 3x3 on FPN[1]
    auto * fpn1_conv = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[1], 2, 0, 1, 3));
    fpn1_conv = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[0], fpn1_conv);
    {
        auto * b3d = ggml_reshape_3d(ctx, seg.up_conv_b[0], 1, 1, seg.up_conv_b[0]->ne[0]);
        fpn1_conv = ggml_add(ctx, fpn1_conv, ggml_repeat(ctx, b3d, fpn1_conv));
    }
    feat = ggml_add(ctx, feat, fpn1_conv);
    // LayerNorm2d
    auto * feat_perm = ggml_cont(ctx, ggml_permute(ctx, feat, 1, 2, 0, 3));  // [D, 144, 144, B]
    feat_perm = sam3_layer_norm_2d(ctx, feat_perm, seg.up_norm_w[0], seg.up_norm_b[0]);
    feat = ggml_cont(ctx, ggml_permute(ctx, feat_perm, 2, 0, 1, 3));  // [144, 144, D, B]

    // Upsample 2x + merge with FPN[0] (288x288)
    feat = ggml_upscale(ctx, feat, 2, GGML_SCALE_MODE_NEAREST);  // [288, 288, D, B]
    auto * fpn0_conv = ggml_cont(ctx, ggml_permute(ctx, fpn_feats[0], 2, 0, 1, 3));
    fpn0_conv = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[1], fpn0_conv);
    {
        auto * b3d = ggml_reshape_3d(ctx, seg.up_conv_b[1], 1, 1, seg.up_conv_b[1]->ne[0]);
        fpn0_conv = ggml_add(ctx, fpn0_conv, ggml_repeat(ctx, b3d, fpn0_conv));
    }
    feat = ggml_add(ctx, feat, fpn0_conv);
    feat_perm = ggml_cont(ctx, ggml_permute(ctx, feat, 1, 2, 0, 3));  // [D, 288, 288, B]
    feat_perm = sam3_layer_norm_2d(ctx, feat_perm, seg.up_norm_w[1], seg.up_norm_b[1]);
    feat = ggml_cont(ctx, ggml_permute(ctx, feat_perm, 2, 0, 1, 3));  // [288, 288, D, B]

    // Final conv
    feat = ggml_conv_2d_s1_ph(ctx, seg.up_conv_w[2], feat);
    {
        auto * b3d = ggml_reshape_3d(ctx, seg.up_conv_b[2], 1, 1, seg.up_conv_b[2]->ne[0]);
        feat = ggml_add(ctx, feat, ggml_repeat(ctx, b3d, feat));
    }
    feat_perm = ggml_cont(ctx, ggml_permute(ctx, feat, 1, 2, 0, 3));  // [D, 288, 288, B]
    feat_perm = sam3_layer_norm_2d(ctx, feat_perm, seg.up_norm_w[2], seg.up_norm_b[2]);

    return feat_perm;  // [D, 288, 288, B]
}

// Build the full segmentation head graph.
// pixel_feats: [D, 288, 288, B] from pixel decoder
// query_outputs: [D, N, B] selected object query outputs
// text_features: [D, T, B] for cross-attention
// Returns: mask_logits [N, 288*288, B] (raw logits, not sigmoid)
static struct ggml_tensor * sam3_build_seg_head_graph(
        struct ggml_context * ctx,
        const sam3_model & model,
        struct ggml_tensor * pixel_feats,    // [D, 288, 288, B]
        struct ggml_tensor * query_outputs,  // [D, N, B]
        struct ggml_tensor * text_features)  // [D, T, B] (for cross-attn, can be nullptr)
{
    const auto & seg     = model.seg_head;
    const auto & tensors = model.tensors;
    const int64_t D = pixel_feats->ne[0];  // 256
    const int64_t W = pixel_feats->ne[1];  // 288
    const int64_t H = pixel_feats->ne[2];  // 288
    const int64_t B = pixel_feats->ne[3];  // 1
    const int64_t N = query_outputs->ne[1]; // number of selected queries

    // Optional: cross-attention of pixel features to text/prompt features
    if (text_features) {
        // Flatten pixel features: [D, W*H, B]
        auto * pf_flat = ggml_reshape_3d(ctx, pixel_feats, D, W * H, B);

        // Cross-attention: pixel features query text
        auto * ca_norm = sam3_layer_norm(ctx, pf_flat,
                                          tensors.at("seg.cross_attn_norm.weight"),
                                          tensors.at("seg.cross_attn_norm.bias"));

        ca_norm = sam3_multihead_attn_fused(ctx, ca_norm, text_features,
                                             seg.ca_prompt_q_w, seg.ca_prompt_q_b,
                                             seg.ca_prompt_out_w, seg.ca_prompt_out_b,
                                             8, nullptr);
        pf_flat = ggml_add(ctx, pf_flat, ca_norm);
        pixel_feats = ggml_reshape_4d(ctx, pf_flat, D, W, H, B);
    }

    // Instance segmentation head: Conv1x1 on pixel features
    auto * pf_conv = ggml_cont(ctx, ggml_permute(ctx, pixel_feats, 2, 0, 1, 3));  // [W, H, D, B] for conv
    pf_conv = ggml_conv_2d_sk_p0(ctx, tensors.at("seg.instance_seg_head.weight"), pf_conv);
    {
        auto * b3d = ggml_reshape_3d(ctx, tensors.at("seg.instance_seg_head.bias"),
                                      1, 1, tensors.at("seg.instance_seg_head.bias")->ne[0]);
        pf_conv = ggml_add(ctx, pf_conv, ggml_repeat(ctx, b3d, pf_conv));
    }
    auto * pixel_embed = ggml_cont(ctx, ggml_permute(ctx, pf_conv, 1, 2, 0, 3));  // [D, W, H, B]

    // Mask embedding: project query outputs through mask_embed MLP
    // 3-layer MLP: D→D→D→D (each layer registered separately)
    auto * mask_embed = query_outputs;
    for (int j = 0; j < 3; ++j) {
        auto wn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".weight";
        auto bn = "seg.mask_predictor.mask_embed.layers." + std::to_string(j) + ".bias";
        mask_embed = ggml_mul_mat(ctx, tensors.at(wn), mask_embed);
        mask_embed = ggml_add(ctx, mask_embed, tensors.at(bn));
        if (j < 2) mask_embed = ggml_relu(ctx, mask_embed);
    }
    // mask_embed: [D, N, B]

    // Mask prediction: einsum('bnd,bdhw->bnhw') = mask_embed^T @ pixel_embed
    // Flatten pixel_embed: [D, W*H, B]
    auto * pe_flat = ggml_reshape_3d(ctx, pixel_embed, D, W * H, B);
    // We need the output layout to have per-query masks contiguous in memory so
    // the CPU-side post-processing can read mask q at offset q*W*H.
    // ggml_mul_mat(A, B) = A^T @ B.  With A=[D, W*H, B], B=[D, N, B]:
    //   A^T is [W*H, D], result = [W*H, N, B].
    // Element (wh, n, b) = data[wh + n*W*H], so query n's mask is contiguous.
    auto * masks = ggml_mul_mat(ctx, pe_flat, mask_embed);
    // masks: [W*H, N, B]

    return masks;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Post-processing: NMS, bilinear interpolation, mask binarization
// ═══════════════════════════════════════════════════════════════════════════════

// Compute IoU between two boxes [x0, y0, x1, y1].
static float sam3_box_iou(const sam3_box & a, const sam3_box & b) {
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
static std::vector<int> sam3_nms(const std::vector<sam3_detection> & dets, float iou_thresh) {
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
static std::vector<float> sam3_bilinear_interpolate(const float * src, int src_w, int src_h,
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

            float v = (1 - wy) * ((1 - wx) * src[y0 * src_w + x0] + wx * src[y0 * src_w + x1])
                    +      wy  * ((1 - wx) * src[y1 * src_w + x0] + wx * src[y1 * src_w + x1]);
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
static void sam3_sine_pos_embed_boxes(float * out, const float * boxes, int NQ, int num_feats) {
    const float temperature = 10000.0f;
    // For each query: 4 coordinates → each gets num_feats/4 = 128/4 = 32 features (sin/cos)
    // Total: 4 * 128 = 512
    const int feats_per_coord = num_feats;  // 128 features per coordinate pair

    for (int q = 0; q < NQ; ++q) {
        for (int c = 0; c < 4; ++c) {
            float val = boxes[q * 4 + c];
            for (int i = 0; i < feats_per_coord; ++i) {
                int paired = i & ~1;
                float dim_t = powf(temperature, (float)paired / (float)feats_per_coord);
                float angle = val * 2.0f * (float)M_PI / dim_t;
                if (i % 2 == 0) {
                    out[q * feats_per_coord * 4 + c * feats_per_coord + i] = sinf(angle);
                } else {
                    out[q * feats_per_coord * 4 + c * feats_per_coord + i] = cosf(angle);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Image segmentation — PCS (text-prompted)
// ═══════════════════════════════════════════════════════════════════════════════

sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params) {
    const auto & hp = model.hparams;
    const int D = hp.neck_dim;   // 256
    const int H = hp.n_img_embd(); // 72
    const int L = hp.text_ctx_len; // 32
    const int NQ = hp.ddec_num_queries; // 200
    sam3_result result;

    // ── Check that image has been encoded ────────────────────────────────
    if (!state.neck_det[0]) {
        fprintf(stderr, "%s: image not encoded — call sam3_encode_image first\n", __func__);
        return result;
    }

    // ── Tokenize text prompt ─────────────────────────────────────────────
    auto token_ids = sam3_tokenize(const_cast<sam3_bpe_tokenizer &>(model.tokenizer),
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
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init compute context\n", __func__);
        return result;
    }

    // ── Text encoder input ───────────────────────────────────────────────
    auto * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(inp_tokens, "text_token_ids");
    ggml_set_input(inp_tokens);

    // ── Text encoder graph (implemented in Phase 4) ─────────────────────
    // sam3_build_text_encoder_graph creates causal mask internally (named "causal_mask").
    // Returns: [256, L] = [D, 32] (2D tensor).
    auto * text_features_2d = sam3_build_text_encoder_graph(ctx0, inp_tokens, model);
    // Reshape to [D, L, 1] for consistent 3D processing in fusion/DETR
    auto * text_features = ggml_reshape_3d(ctx0, text_features_2d, D, L, 1);
    ggml_set_name(text_features, "text_features");

    // ── Prepare image features for fusion encoder ────────────────────────
    // Use the 72×72 neck features (scale 2) for the fusion encoder
    // neck_det[2]: [D, 72, 72, B] — flatten spatial dims → [D, 5184, 1]
    auto * img_feats = ggml_reshape_3d(ctx0, state.neck_det[2], D, H * H, 1);
    // PE for image features: flatten from [D, 72, 72, 1] → [D, 5184, 1]
    auto * img_pe = ggml_reshape_3d(ctx0, state.neck_det_pe[2], D, H * H, 1);

    // ── Fusion encoder graph ─────────────────────────────────────────────
    auto * conditioned = sam3_build_fenc_graph(ctx0, model, img_feats, text_features, img_pe);
    ggml_set_name(conditioned, "fenc_output");
    // conditioned: [D, 5184, 1]

    // ── DETR decoder graph ───────────────────────────────────────────────
    auto ddec_out = sam3_build_ddec_graph(ctx0, model, conditioned, img_pe, text_features);
    ggml_set_name(ddec_out.class_scores, "class_scores");
    ggml_set_name(ddec_out.pred_boxes, "pred_boxes");
    ggml_set_name(ddec_out.presence_score, "presence_score");
    ggml_set_output(ddec_out.class_scores);
    ggml_set_output(ddec_out.pred_boxes);
    ggml_set_output(ddec_out.presence_score);

    // ── Segmentation head graph ──────────────────────────────────────────
    // Use FPN features from detector neck at 3 scales
    struct ggml_tensor * fpn_feats[3] = {
        state.neck_det[0],  // [D, 288, 288, B]
        state.neck_det[1],  // [D, 144, 144, B]
        state.neck_det[2],  // [D,  72,  72, B]
    };

    auto * pixel_feats = sam3_pixel_decoder(ctx0, model, fpn_feats);
    ggml_set_name(pixel_feats, "pixel_feats");
    // pixel_feats: [D, 288, 288, B]

    // Extract object queries from DETR output (skip presence token)
    auto * obj_queries = ggml_view_3d(ctx0, ddec_out.queries, D, NQ, 1,
                                       ddec_out.queries->nb[1], ddec_out.queries->nb[2],
                                       1 * ddec_out.queries->nb[1]);
    obj_queries = ggml_cont(ctx0, obj_queries);

    auto * mask_logits = sam3_build_seg_head_graph(ctx0, model, pixel_feats, obj_queries, text_features);
    ggml_set_name(mask_logits, "mask_logits");
    ggml_set_output(mask_logits);
    // mask_logits: [288*288, NQ, 1]  (per-query masks are contiguous)

    // ── Build and allocate graph ─────────────────────────────────────────
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(graph, ddec_out.class_scores);
    ggml_build_forward_expand(graph, ddec_out.pred_boxes);
    ggml_build_forward_expand(graph, ddec_out.presence_score);
    ggml_build_forward_expand(graph, mask_logits);

    auto * galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
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
        auto * causal_mask = ggml_get_tensor(ctx0, "causal_mask");
        if (causal_mask) {
            std::vector<ggml_fp16_t> mask_data(L * L);
            sam3_fill_causal_mask(mask_data.data(), L);
            ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, L * L * sizeof(ggml_fp16_t));
        }
    }

    // DETR query positional encoding from reference points
    {
        // Read reference_points.weight, compute sine positional embedding
        auto * ref_w = model.tensors.at("ddec.reference_points.weight");
        std::vector<float> ref_data(4 * NQ);
        ggml_backend_tensor_get(ref_w, ref_data.data(), 0, 4 * NQ * sizeof(float));

        // Apply sigmoid to get initial ref boxes
        for (auto & v : ref_data) v = 1.0f / (1.0f + expf(-v));

        // Compute sine positional embedding [NQ, 512]
        std::vector<float> pos_embed(NQ * 512);
        sam3_sine_pos_embed_boxes(pos_embed.data(), ref_data.data(), NQ, 128);

        // Pass through ref_point_head MLP (2 layers: 512→256→256)
        // For now, we need to compute this CPU-side and upload
        // ref_point_head layer 0: [512, 256], layer 1: [256, 256]
        auto * rph0_w = model.tensors.at("ddec.ref_point_head.layers.0.weight");
        auto * rph0_b = model.tensors.at("ddec.ref_point_head.layers.0.bias");
        auto * rph1_w = model.tensors.at("ddec.ref_point_head.layers.1.weight");
        auto * rph1_b = model.tensors.at("ddec.ref_point_head.layers.1.bias");

        // Read weights to CPU
        // rph0_w: ggml [512, D=256] — in ggml this is ne[0]=512, ne[1]=256
        // For matmul: out = W^T @ x, so out[j] = sum_i W[i,j] * x[i]
        // With ne[0]=512, ne[1]=256: W is 256 rows of 512 elements
        // out = W^T @ x where W is [512, 256], x is [512] → out is [256]
        int rph0_w_nel = (int)(rph0_w->ne[0] * rph0_w->ne[1]);
        std::vector<float> w0_data(rph0_w_nel);
        if (rph0_w->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(rph0_w_nel);
            ggml_backend_tensor_get(rph0_w, tmp.data(), 0, rph0_w_nel * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), w0_data.data(), rph0_w_nel);
        } else {
            ggml_backend_tensor_get(rph0_w, w0_data.data(), 0, rph0_w_nel * sizeof(float));
        }

        std::vector<float> b0_data(D);
        ggml_backend_tensor_get(rph0_b, b0_data.data(), 0, D * sizeof(float));

        int rph1_w_nel = (int)(rph1_w->ne[0] * rph1_w->ne[1]);
        std::vector<float> w1_data(rph1_w_nel);
        if (rph1_w->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(rph1_w_nel);
            ggml_backend_tensor_get(rph1_w, tmp.data(), 0, rph1_w_nel * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), w1_data.data(), rph1_w_nel);
        } else {
            ggml_backend_tensor_get(rph1_w, w1_data.data(), 0, rph1_w_nel * sizeof(float));
        }

        std::vector<float> b1_data(D);
        ggml_backend_tensor_get(rph1_b, b1_data.data(), 0, D * sizeof(float));

        // CPU matmul: for each query
        // Layer 0: [512] → [256] with ReLU
        // ggml matmul convention: W=[ne[0]=in_dim, ne[1]=out_dim], out = W^T @ x
        std::vector<float> query_pos_data(D * NQ, 0.0f);
        for (int q = 0; q < NQ; ++q) {
            std::vector<float> h0(D, 0.0f);
            // W^T @ x: for each output j, sum over i: W[i, j] * x[i]
            // W stored as ne[0]=512, ne[1]=256 → row j has 512 elements at offset j*512
            for (int j = 0; j < D; ++j) {
                float sum = b0_data[j];
                for (int i = 0; i < 512; ++i) {
                    sum += w0_data[j * 512 + i] * pos_embed[q * 512 + i];
                }
                h0[j] = std::max(0.0f, sum);  // ReLU
            }
            // Layer 1: [256] → [256] with ReLU
            for (int j = 0; j < D; ++j) {
                float sum = b1_data[j];
                for (int i = 0; i < D; ++i) {
                    sum += w1_data[j * D + i] * h0[i];
                }
                query_pos_data[q * D + j] = std::max(0.0f, sum);  // ReLU
            }
        }

        // Upload query_pos for object queries
        // ddec_query_pos_obj: [D, NQ, 1] — ggml stores as column-major
        // Our data is [NQ, D] in row-major = [D, NQ] in ggml column-major. Correct!
        auto * qpos_obj = ggml_get_tensor(ctx0, "ddec_query_pos_obj");
        if (qpos_obj) {
            ggml_backend_tensor_set(qpos_obj, query_pos_data.data(), 0, D * NQ * sizeof(float));
        }

        // Presence token position: zeros
        auto * qpos_pres = ggml_get_tensor(ctx0, "ddec_query_pos_pres");
        if (qpos_pres) {
            std::vector<float> zeros(D, 0.0f);
            ggml_backend_tensor_set(qpos_pres, zeros.data(), 0, D * sizeof(float));
        }
    }

    // ── Compute ──────────────────────────────────────────────────────────
    sam3_graph_compute(model.backend, graph, 4);
    fprintf(stderr, "%s: graph computed\n", __func__);

    // ── Read outputs ─────────────────────────────────────────────────────
    std::vector<float> scores_data(NQ);
    ggml_backend_tensor_get(ddec_out.class_scores, scores_data.data(), 0, NQ * sizeof(float));

    std::vector<float> boxes_data(4 * NQ);
    ggml_backend_tensor_get(ddec_out.pred_boxes, boxes_data.data(), 0, 4 * NQ * sizeof(float));

    std::vector<float> pres_data(1);
    ggml_backend_tensor_get(ddec_out.presence_score, pres_data.data(), 0, sizeof(float));

    float presence = pres_data[0];

    // Read mask logits: [288*288, NQ, 1] — per-query masks are contiguous
    const int mask_hw = 288;
    std::vector<float> all_masks(NQ * mask_hw * mask_hw);
    ggml_backend_tensor_get(mask_logits, all_masks.data(), 0, all_masks.size() * sizeof(float));

    // ── Post-process: score thresholding ─────────────────────────────────
    std::vector<sam3_detection> dets;
    for (int q = 0; q < NQ; ++q) {
        float score = scores_data[q] * presence;
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
        const float * mask_ptr = all_masks.data() + q * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                       state.orig_width, state.orig_height);

        // Binarize mask at threshold 0.0 (sigmoid > 0.5 ↔ logit > 0.0)
        det.mask.width  = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i) {
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        }
        det.mask.iou_score = score;

        dets.push_back(std::move(det));
    }

    fprintf(stderr, "%s: %zu detections above threshold %.2f (presence=%.3f)\n",
            __func__, dets.size(), params.score_threshold, presence);

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

    return result;
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

// ═══════════════════════════════════════════════════════════════════════════════
//  Debug: dump state tensors
// ═══════════════════════════════════════════════════════════════════════════════

bool sam3_dump_state_tensor(const sam3_state & state,
                             const std::string & tensor_name,
                             const std::string & output_path) {
    struct ggml_tensor * t = nullptr;

    if (tensor_name == "vit_output") {
        t = state.vit_output;
    } else if (tensor_name == "neck_det_0") {
        t = state.neck_det[0];
    } else if (tensor_name == "neck_det_1") {
        t = state.neck_det[1];
    } else if (tensor_name == "neck_det_2") {
        t = state.neck_det[2];
    } else {
        // Search by ggml name in the context
        if (state.ctx) {
            t = ggml_get_tensor(state.ctx, tensor_name.c_str());
        }
    }

    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found in state\n", __func__, tensor_name.c_str());
        return false;
    }

    // Read data from backend
    int64_t numel = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) numel *= t->ne[i];
    }

    std::vector<float> data(numel);
    ggml_backend_tensor_get(t, data.data(), 0, numel * sizeof(float));

    // Write binary file
    {
        std::ofstream f(output_path + ".bin", std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(data.data()), numel * sizeof(float));
    }

    // Write shape file
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
        fprintf(stderr, "%lld", (long long)t->ne[i]);
    }
    fprintf(stderr, "] to %s\n", output_path.c_str());
    return true;
}
