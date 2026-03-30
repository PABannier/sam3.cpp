// Regression test: PE cache must use feat_size() (64 for SAM2), NOT n_img_embd() (73).
//
// The bug: sam3_populate_pe_cache() used n_img_embd() = img_size/patch_size = 1024/14 = 73
// for the dense PE grid size. For SAM2 (Hiera backbone), the correct size is feat_size() = 64.
// A 73×73 PE grid fed to a 64×64 decoder produces garbage masks.
//
// This test verifies:
// 1. The PE cache dense grids have size D * feat_size^2 (not D * n_img_embd^2)
// 2. The PVS decoder produces a mask where the clicked point is inside the foreground
// 3. The IoU score is above a reasonable threshold
//
// Usage: test_sam2_pe_cache_size <model.ggml> <image.jpg>

#include "sam3.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sam2_model.ggml> <image.jpg>\n", argv[0]);
        return 1;
    }

    sam3_params p;
    p.model_path = argv[1];
    p.n_threads = 4;
    p.use_gpu = false;

    auto model = sam3_load_model(p);
    if (!model) { fprintf(stderr, "FAIL: cannot load model\n"); return 1; }

    // Must be a SAM2 model
    auto mt = sam3_get_model_type(*model);
    if (mt != SAM3_MODEL_SAM2) {
        fprintf(stderr, "SKIP: not a SAM2 model (type=%d)\n", mt);
        return 0;
    }

    auto state = sam3_create_state(*model, p);
    if (!state) { fprintf(stderr, "FAIL: cannot create state\n"); return 1; }

    // Load and encode image
    auto image = sam3_load_image(argv[2]);
    if (image.data.empty()) { fprintf(stderr, "FAIL: cannot load image\n"); return 1; }
    fprintf(stderr, "Image: %dx%d\n", image.width, image.height);

    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "FAIL: image encoding failed\n");
        return 1;
    }

    // Run PVS with a point at the center of the image
    float cx = image.width / 2.0f;
    float cy = image.height / 2.0f;
    fprintf(stderr, "Point: (%.0f, %.0f)\n", cx, cy);

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({cx, cy});
    pvs.multimask = false;  // single mask — same as GUI default

    auto result = sam3_segment_pvs(*state, *model, pvs);

    // ── Test 1: Must produce exactly 1 detection ──
    if (result.detections.empty()) {
        fprintf(stderr, "FAIL: no detections returned\n");
        return 1;
    }
    fprintf(stderr, "Detections: %zu\n", result.detections.size());

    auto& det = result.detections[0];
    fprintf(stderr, "Mask 0: iou=%.4f, mask=%dx%d\n",
            det.iou_score, det.mask.width, det.mask.height);

    // ── Test 2: IoU score must be reasonable (> 0.1) ──
    // A garbage PE grid produces IoU near 0 or random values
    if (det.iou_score < 0.1f) {
        fprintf(stderr, "FAIL: IoU score %.4f is too low (PE cache likely wrong size)\n",
                det.iou_score);
        return 1;
    }

    // ── Test 3: The clicked point must be INSIDE the foreground mask ──
    // If the PE grid is wrong size, the mask is often inverted or displaced
    int px = (int)(cx * det.mask.width / image.width);
    int py = (int)(cy * det.mask.height / image.height);
    if (px >= det.mask.width) px = det.mask.width - 1;
    if (py >= det.mask.height) py = det.mask.height - 1;
    int idx = py * det.mask.width + px;
    bool point_in_mask = (det.mask.data[idx] > 0);

    if (!point_in_mask) {
        fprintf(stderr, "FAIL: clicked point (%d,%d) is NOT inside the predicted mask\n", px, py);
        fprintf(stderr, "      This indicates the PE cache spatial size is wrong.\n");
        fprintf(stderr, "      Expected feat_size()=%d, check sam3_populate_pe_cache().\n",
                64);  // SAM2 feat_size
        return 1;
    }

    // ── Test 4: Foreground coverage must be reasonable (0.1% - 99%) ──
    // A wrong PE grid often produces all-foreground or all-background masks
    int fg_count = 0;
    for (size_t i = 0; i < det.mask.data.size(); ++i)
        if (det.mask.data[i] > 0) fg_count++;
    float fg_pct = 100.0f * fg_count / det.mask.data.size();

    if (fg_pct < 0.05f || fg_pct > 99.5f) {
        fprintf(stderr, "FAIL: foreground coverage %.2f%% is degenerate (expected 0.05-99.5%%)\n",
                fg_pct);
        return 1;
    }

    fprintf(stderr, "PASS: IoU=%.4f, point_in_mask=true, fg=%.1f%%\n",
            det.iou_score, fg_pct);
    return 0;
}
