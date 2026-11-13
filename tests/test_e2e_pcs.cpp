/**
 * End-to-end PCS pipeline test: loads Python-preprocessed image,
 * runs through C++ ViT + neck + text encoder + fusion encoder + DETR decoder
 * + segmentation head, dumps all intermediate tensors for comparison against
 * Python reference.
 *
 * Usage:
 *   ./test_e2e_pcs [--gpu] [--n-threads N] <model.ggml> <ref_dir>
 *
 *   --gpu        request the GPU backend (Metal on macOS; HIP/CUDA when the
 *                ggml build provides it). Falls back to CPU otherwise.
 *   --n-threads  CPU thread count (default 4).
 *
 * ref_dir should contain pcs_ref/ with tensors from dump_e2e_pcs.py:
 *   - preprocessed_chw.bin/.shape
 *   - token_ids.bin/.shape
 *   - text_features.bin/.shape
 *   - neck_det_*.bin/.shape
 *   - fenc_layer*_out.bin/.shape
 *   - ddec_layer*_out.bin/.shape
 *   - pred_logits.bin/.shape, pred_boxes.bin/.shape
 *   - seg_pred_masks.bin/.shape
 *   etc.
 */
#include "sam3.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ── Metric row (same as test_e2e_pvs.cpp) ──

struct metric_row {
    std::string stage;
    std::string name;
    std::string shape_str;
    float mae        = 0.0f;
    float max_abs    = 0.0f;
    float mean_rel   = 0.0f;
    float cosine     = 0.0f;
    float p95        = 0.0f;
    float p99        = 0.0f;
    bool  ok         = false;
    float tolerance  = 0.0f;
};

static metric_row compute_full_metrics(const float * a, const float * b, int n, float tol) {
    metric_row m;
    m.tolerance = tol;
    if (n == 0) return m;

    double sum_diff = 0.0, sum_rel = 0.0;
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;
    float max_d = 0.0f;

    std::vector<float> diffs(n);
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        diffs[i] = d;
        sum_diff += d;
        if (d > max_d) max_d = d;
        float denom = fabsf(b[i]) + 1e-8f;
        sum_rel += d / denom;
        dot_ab += (double)a[i] * (double)b[i];
        dot_aa += (double)a[i] * (double)a[i];
        dot_bb += (double)b[i] * (double)b[i];
    }

    m.mae = (float)(sum_diff / n);
    m.max_abs = max_d;
    m.mean_rel = (float)(sum_rel / n);
    double ddenom = sqrt(dot_aa) * sqrt(dot_bb);
    m.cosine = ddenom > 0.0 ? (float)(dot_ab / ddenom) : 0.0f;

    std::sort(diffs.begin(), diffs.end());
    m.p95 = diffs[(int)(0.95 * n)];
    m.p99 = diffs[(int)(0.99 * n)];
    m.ok = max_d <= tol;
    return m;
}

// Compare raw flat data (same byte layout between Python and C++)
static metric_row compare_flat(const char * stage, const char * label,
                                const std::string & cpp_path,
                                const std::string & ref_path,
                                float atol) {
    auto cpp = load_ref_f32(cpp_path);
    auto ref = load_ref_f32(ref_path);
    metric_row m;
    m.stage = stage;
    m.name = label;
    m.tolerance = atol;
    if (cpp.data.empty() || ref.data.empty()) {
        fprintf(stderr, "  [SKIP] %s (cpp=%s ref=%s)\n", label,
                cpp.data.empty() ? "missing" : "ok",
                ref.data.empty() ? "missing" : "ok");
        return m;
    }
    std::string s;
    for (size_t i = 0; i < ref.shape.size(); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(ref.shape[i]);
    }
    m.shape_str = s;
    int n = std::min(cpp.numel(), ref.numel());
    return compute_full_metrics(cpp.data.data(), ref.data.data(), n, atol);
}

static double now_ms() {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char ** argv) {
    bool use_gpu = false;
    int n_threads = 4;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--n-threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!arg.empty() && arg[0] == '-') {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() < 2) {
        fprintf(stderr, "Usage: %s [--gpu] [--n-threads N] <model.ggml> <ref_dir>\n", argv[0]);
        fprintf(stderr, "\nref_dir should contain pcs_ref/ from dump_e2e_pcs.py\n");
        return 1;
    }

    const std::string model_path = positional[0];
    const std::string ref_dir = positional[1] + "/pcs_ref";
    const std::string cpp_out = positional[1] + "/pcs_cpp";
    ensure_dir(cpp_out);

    // ═══ Load model ═══
    fprintf(stderr, "\n═══ Loading model ═══\n");
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = use_gpu;
    params.n_threads = n_threads;
    fprintf(stderr, "  use_gpu = %d, n_threads = %d\n", use_gpu, n_threads);

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    // ═══ Stage 1: Load Python-preprocessed image ═══
    fprintf(stderr, "\n═══ Stage 1: Loading Python-preprocessed image ═══\n");
    auto ref_img = load_ref_f32(ref_dir + "/preprocessed_chw");
    if (ref_img.data.empty()) {
        fprintf(stderr, "Failed to load %s/preprocessed_chw.bin\n", ref_dir.c_str());
        return 1;
    }
    fprintf(stderr, "  Loaded preprocessed image: %d elements\n", ref_img.numel());

    const int img_size = 1008;
    double t_encode0 = now_ms();
    bool ok = sam3_encode_image_from_preprocessed(*state, *model, ref_img.data.data(), img_size);
    double t_encode = now_ms() - t_encode0;
    if (!ok) {
        fprintf(stderr, "sam3_encode_image_from_preprocessed failed!\n");
        return 1;
    }
    fprintf(stderr, "  Image encoded successfully (%.1f ms)\n", t_encode);

    // ═══ Stage 2: Dump ViT block outputs (from state debug tensors) ═══
    fprintf(stderr, "\n═══ Stage 2: ViT Backbone ═══\n");
    std::vector<metric_row> report;

    // ViT block outputs are stored in state during encoding
    for (int i : {0, 7, 15, 23, 31}) {
        char cpp_name[64], ref_name[64];
        snprintf(cpp_name, sizeof(cpp_name), "dbg_block_%d_out", i);
        snprintf(ref_name, sizeof(ref_name), "vit_block_%02d_out", i);

        sam3_dump_state_tensor(*state, cpp_name, cpp_out + "/" + cpp_name);
        auto cpp = load_ref_f32(cpp_out + "/" + cpp_name);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);
        if (cpp.data.empty() || ref.data.empty()) {
            fprintf(stderr, "  [SKIP] %s\n", ref_name);
            continue;
        }
        // f32 accumulation through 32 blocks: allow up to 0.07 max error
        int n = std::min(cpp.numel(), ref.numel());
        auto m = compute_full_metrics(cpp.data.data(), ref.data.data(), n, 0.07f);
        m.stage = "ViT";
        m.name = ref_name;
        char shape_s[64];
        snprintf(shape_s, sizeof(shape_s), "%d,%d,%d,%d",
                 ref.shape.size()>0?ref.shape[0]:0, ref.shape.size()>1?ref.shape[1]:0,
                 ref.shape.size()>2?ref.shape[2]:0, ref.shape.size()>3?ref.shape[3]:0);
        m.shape_str = shape_s;
        report.push_back(m);
    }

    // ═══ Stage 3: Neck ═══
    fprintf(stderr, "\n═══ Stage 3: Neck (SimpleFPN) ═══\n");
    // Dump neck detector features — these are stored in state
    for (int i = 0; i < 3; ++i) {
        char tensor_name[32], ref_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "neck_det_%d", i);
        snprintf(ref_name, sizeof(ref_name), "neck_det_%d", i);

        sam3_dump_state_tensor(*state, tensor_name, cpp_out + "/" + tensor_name);
        auto cpp_t = load_ref_f32(cpp_out + "/" + tensor_name);
        auto ref_t = load_ref_f32(ref_dir + "/" + ref_name);
        if (cpp_t.data.empty() || ref_t.data.empty()) {
            fprintf(stderr, "  [SKIP] %s\n", ref_name);
            continue;
        }

        // ggml stores [C, W, H, B], Python dumps NCHW as [C, W, H, B] via ggml layout
        // Both should match in flat layout now
        int n = std::min(cpp_t.numel(), ref_t.numel());
        auto m = compute_full_metrics(cpp_t.data.data(), ref_t.data.data(), n, 5e-3f);
        m.stage = "Neck";
        m.name = ref_name;
        std::string s;
        for (size_t j = 0; j < ref_t.shape.size(); ++j) {
            if (j > 0) s += ",";
            s += std::to_string(ref_t.shape[j]);
        }
        m.shape_str = s;
        report.push_back(m);
    }

    // ═══ Stage 4-9: Text + Fusion + DETR + Segmentation (via sam3_test_dump_phase5) ═══
    fprintf(stderr, "\n═══ Stage 4-9: Text Encoder + Fusion + DETR + Segmentation ═══\n");

    // Load reference token IDs (saved as float32 by Python dump)
    auto ref_tokens_f = load_ref_f32(ref_dir + "/token_ids");
    if (ref_tokens_f.data.empty()) {
        fprintf(stderr, "Failed to load token IDs from ref_dir\n");
        return 1;
    }
    // Convert float→int32
    std::vector<int32_t> token_ids(ref_tokens_f.numel());
    for (int i = 0; i < ref_tokens_f.numel(); ++i) {
        token_ids[i] = (int32_t)ref_tokens_f.data[i];
    }
    // Pad to 32 if needed
    while ((int)token_ids.size() < 32) token_ids.push_back(0);
    fprintf(stderr, "  Token IDs: [");
    for (int i = 0; i < 32; ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", token_ids[i]);
    }
    fprintf(stderr, "]\n");

    // Run the full phase 5 dump (text encoder + fusion encoder + DETR decoder + seg head)
    const std::string phase5_out = cpp_out + "/phase5";
    ensure_dir(phase5_out);
    double t_phase5_0 = now_ms();
    if (!sam3_test_dump_phase5(*model, *state, token_ids, phase5_out, params.n_threads)) {
        fprintf(stderr, "  [FAIL] sam3_test_dump_phase5 failed\n");
        return 1;
    }
    double t_phase5 = now_ms() - t_phase5_0;
    fprintf(stderr, "  Phase 5 tensors dumped to %s (%.1f ms)\n", phase5_out.c_str(), t_phase5);

    // ═══ Compare text encoder ═══
    fprintf(stderr, "\n  --- Text Encoder ---\n");
    {
        auto m = compare_flat("TextEnc", "text_features",
                               phase5_out + "/text_features",
                               ref_dir + "/text_features", 1e-4f);
        m.stage = "TextEnc";
        report.push_back(m);
    }

    // ═══ Compare fusion encoder layers ═══
    fprintf(stderr, "\n  --- Fusion Encoder ---\n");
    {
        // Compare fenc_output (the final encoder output)
        // C++ dumps as [D, 5184, 1], Python dumps as [5184, 1, 256]
        // Both are flat [5184*256] floats in memory
        auto m = compare_flat("FEnc", "fenc_output",
                               phase5_out + "/fenc_output",
                               ref_dir + "/fenc_output", 5e-3f);
        m.stage = "FEnc";
        report.push_back(m);
    }
    for (int i = 0; i < 6; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        auto m = compare_flat("FEnc", name,
                               phase5_out + "/" + name,
                               ref_dir + "/" + name, 5e-3f);
        m.stage = "FEnc";
        report.push_back(m);
    }

    // ═══ Compare DETR decoder layers ═══
    fprintf(stderr, "\n  --- DETR Decoder ---\n");
    for (int i = 0; i < 6; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "ddec_layer%d_out", i);
        auto m = compare_flat("DDec", name,
                               phase5_out + "/" + name,
                               ref_dir + "/" + name, 0.1f);
        m.stage = "DDec";
        report.push_back(m);
    }

    // ═══ Compare scoring ═══
    fprintf(stderr, "\n  --- Scoring ---\n");
    {
        auto m = compare_flat("Score", "scoring_class_scores",
                               phase5_out + "/scoring_class_scores",
                               ref_dir + "/scoring_output", 0.5f);
        m.stage = "Score";
        report.push_back(m);
    }

    // ═══ Compare segmentation masks ═══
    fprintf(stderr, "\n  --- Segmentation ---\n");
    {
        auto m = compare_flat("SegHead", "seg_mask_logits",
                               phase5_out + "/seg_mask_logits",
                               ref_dir + "/seg_pred_masks", 1.0f);
        m.stage = "SegHead";
        report.push_back(m);
    }

    // ═══ Compare final model outputs ═══
    fprintf(stderr, "\n  --- Final Outputs ---\n");
    {
        // pred_boxes
        auto m = compare_flat("Output", "pred_boxes",
                               phase5_out + "/ddec_pred_boxes",
                               ref_dir + "/pred_boxes", 0.1f);
        m.stage = "Output";
        report.push_back(m);
    }
    {
        // presence score
        auto m = compare_flat("Output", "presence_logit",
                               phase5_out + "/ddec_presence_logit",
                               ref_dir + "/presence_logit", 0.5f);
        m.stage = "Output";
        report.push_back(m);
    }

    // ═══ Now run the actual sam3_segment_pcs and compare final results ═══
    fprintf(stderr, "\n═══ Running sam3_segment_pcs (end-to-end) ═══\n");

    // Re-encode image (sam3_test_dump_phase5 may have invalidated state)
    double t_e2e_0 = now_ms();
    ok = sam3_encode_image_from_preprocessed(*state, *model, ref_img.data.data(), img_size);
    if (!ok) {
        fprintf(stderr, "Failed to re-encode image\n");
        return 1;
    }

    sam3_pcs_params pcs;
    pcs.text_prompt = "cat";
    pcs.score_threshold = 0.3f;
    pcs.nms_threshold = 0.1f;

    sam3_result result = sam3_segment_pcs(*state, *model, pcs);
    double t_e2e = now_ms() - t_e2e_0;
    fprintf(stderr, "  sam3_segment_pcs total (re-encode + detect): %.1f ms\n", t_e2e);

    fprintf(stderr, "\n  C++ detections: %zu\n", result.detections.size());
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto & det = result.detections[i];
        fprintf(stderr, "  Det %zu: score=%.4f box=[%.1f, %.1f, %.1f, %.1f]\n",
                i, det.score, det.box.x0, det.box.y0, det.box.x1, det.box.y1);
    }

    // Compare with Python results
    auto py_scores = load_ref_f32(ref_dir + "/final_scores");
    auto py_boxes = load_ref_f32(ref_dir + "/final_boxes_xyxy");

    if (!py_scores.data.empty() && !result.detections.empty()) {
        fprintf(stderr, "\n  Python detections: %d\n", py_scores.numel());
        for (int i = 0; i < py_scores.numel(); ++i) {
            fprintf(stderr, "  Py Det %d: score=%.4f box=[%.1f, %.1f, %.1f, %.1f]\n",
                    i, py_scores.data[i],
                    py_boxes.data[i*4+0], py_boxes.data[i*4+1],
                    py_boxes.data[i*4+2], py_boxes.data[i*4+3]);
        }

        // Compare top detection
        float score_diff = fabsf(result.detections[0].score - py_scores.data[0]);
        float box_diff = fabsf(result.detections[0].box.x0 - py_boxes.data[0])
                       + fabsf(result.detections[0].box.y0 - py_boxes.data[1])
                       + fabsf(result.detections[0].box.x1 - py_boxes.data[2])
                       + fabsf(result.detections[0].box.y1 - py_boxes.data[3]);
        fprintf(stderr, "\n  Score diff: %.4f\n", score_diff);
        fprintf(stderr, "  Box L1 diff: %.2f pixels\n", box_diff);

        metric_row m_score;
        m_score.stage = "E2E";
        m_score.name = "score_match";
        m_score.max_abs = score_diff;
        m_score.ok = score_diff < 0.05f;
        m_score.tolerance = 0.05f;
        report.push_back(m_score);

        metric_row m_box;
        m_box.stage = "E2E";
        m_box.name = "box_match (L1 px)";
        m_box.max_abs = box_diff;
        m_box.ok = box_diff < 50.0f;
        m_box.tolerance = 50.0f;
        report.push_back(m_box);
    }

    // ═══ Compare masks ═══
    if (!result.detections.empty() && !result.detections[0].mask.data.empty()) {
        auto py_mask = load_ref_f32(ref_dir + "/final_masks_binary");
        if (!py_mask.data.empty()) {
            int mask_w = result.detections[0].mask.width;
            int mask_h = result.detections[0].mask.height;
            int n_pix = mask_w * mask_h;
            int n_match = 0;
            int n_total = std::min(n_pix, py_mask.numel());
            for (int i = 0; i < n_total; ++i) {
                float cpp_val = result.detections[0].mask.data[i] > 0 ? 1.0f : 0.0f;
                float py_val = py_mask.data[i];
                if (fabsf(cpp_val - py_val) < 0.5f) n_match++;
            }
            float iou_approx = (float)n_match / (float)n_total;
            fprintf(stderr, "\n  Mask agreement: %d/%d pixels (%.2f%%)\n",
                    n_match, n_total, 100.0f * iou_approx);

            metric_row m_mask;
            m_mask.stage = "E2E";
            m_mask.name = "mask_agreement";
            m_mask.max_abs = 1.0f - iou_approx;
            m_mask.ok = iou_approx > 0.95f;
            m_mask.tolerance = 0.05f;
            m_mask.cosine = iou_approx;
            report.push_back(m_mask);
        }
    }

    // ═══ Print Full Report ═══
    fprintf(stderr, "\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  END-TO-END PCS PIPELINE COMPARISON: Python vs C++ (cat.jpg with text=\"cat\")\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    int n_pass = 0, n_fail = 0, n_skip = 0;
    fprintf(stderr, "  %-8s %-8s %-30s %-20s %12s %12s %12s %12s %12s %12s\n",
            "Status", "Stage", "Tensor", "Shape", "MAE", "Max", "RelErr", "Cosine", "P95", "P99");
    fprintf(stderr, "  %-8s %-8s %-30s %-20s %12s %12s %12s %12s %12s %12s\n",
            "------", "-----", "------", "-----", "---", "---", "------", "------", "---", "---");

    for (const auto & m : report) {
        if (m.shape_str.empty() && m.mae == 0.0f && m.max_abs == 0.0f && m.cosine == 0.0f) {
            fprintf(stderr, "  %-8s %-8s %-30s\n", "SKIP", m.stage.c_str(), m.name.c_str());
            n_skip++;
            continue;
        }
        const char * status = m.ok ? "PASS" : "FAIL";
        fprintf(stderr, "  %-8s %-8s %-30s %-20s %12.4e %12.4e %12.4e %12.8f %12.4e %12.4e\n",
                status, m.stage.c_str(), m.name.c_str(), m.shape_str.c_str(),
                m.mae, m.max_abs, m.mean_rel, m.cosine, m.p95, m.p99);
        if (m.ok) n_pass++;
        else n_fail++;
    }

    fprintf(stderr, "\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  TOTAL: %d PASS, %d FAIL, %d SKIP out of %d tensors\n",
            n_pass, n_fail, n_skip, (int)report.size());
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    fprintf(stderr, "═══ PERFORMANCE SUMMARY (%s) ═══\n", use_gpu ? "GPU requested" : "CPU");
    fprintf(stderr, "  Stage 1 (encode ViT + neck):              %10.1f ms\n", t_encode);
    fprintf(stderr, "  Stage 4-9 (text + fenc + ddec + seg dump):%10.1f ms\n", t_phase5);
    fprintf(stderr, "  E2E (re-encode + sam3_segment_pcs):       %10.1f ms\n", t_e2e);

    // Save mask for visual comparison
    if (!result.detections.empty()) {
        std::string mask_path = cpp_out + "/cat_mask.png";
        if (sam3_save_mask(result.detections[0].mask, mask_path)) {
            fprintf(stderr, "  Saved C++ mask: %s\n", mask_path.c_str());
        }
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();

    return n_fail > 0 ? 1 : 0;
}
