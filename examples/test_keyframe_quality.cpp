/**
 * test_keyframe_quality — compare mask quality at different keyframe intervals.
 *
 * Runs video tracking twice per K value: once with K=0 (baseline, full pipeline
 * every frame) and once with keyframe_interval=K. Reports per-frame IoU between
 * the two, plus bounding-box error statistics.
 *
 * Usage:
 *   test_keyframe_quality --model <path> --video <path> [options]
 *
 * Options:
 *   --model <path>         Model .ggml file (required)
 *   --video <path>         Video file (default: data/test_video.mp4)
 *   --point-x <f>          Click X (default: 315.0)
 *   --point-y <f>          Click Y (default: 250.0)
 *   --n-frames <n>         Frames to track (default: 30)
 *   --n-threads <n>        CPU threads (default: 4)
 *   --no-gpu               Use CPU only
 *   --k-values <list>      Comma-separated K values (default: 5,10,15,20,30)
 */

#include "sam3.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>

struct FrameMask {
    std::vector<uint8_t> data; // binary mask (0/255)
    int width  = 0;
    int height = 0;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0; // bounding box
    float score = 0;
    bool valid = false;
};

static float mask_iou(const FrameMask & a, const FrameMask & b) {
    if (!a.valid || !b.valid) return 0.0f;
    if (a.width != b.width || a.height != b.height) return 0.0f;
    int inter = 0, uni = 0;
    for (int i = 0; i < (int)a.data.size(); i++) {
        bool fa = a.data[i] > 127;
        bool fb = b.data[i] > 127;
        if (fa && fb) inter++;
        if (fa || fb) uni++;
    }
    return uni > 0 ? (float)inter / uni : 1.0f;
}

static float box_iou(const FrameMask & a, const FrameMask & b) {
    if (!a.valid || !b.valid) return 0.0f;
    float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    float iw = std::max(0.0f, ix1 - ix0), ih = std::max(0.0f, iy1 - iy0);
    float inter = iw * ih;
    float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);
    float uni = area_a + area_b - inter;
    return uni > 0 ? inter / uni : 1.0f;
}

// Run tracking and collect per-frame masks
static std::vector<FrameMask> run_tracking(
        const sam3_model & model,
        const std::vector<sam3_image> & frames,
        const sam3_params & params,
        float px, float py,
        int keyframe_interval) {
    int n_frames = (int)frames.size();
    std::vector<FrameMask> result(n_frames);

    auto state = sam3_create_state(model, params);
    if (!state) { fprintf(stderr, "ERROR: state creation failed\n"); return result; }

    sam3_visual_track_params vtp;
    vtp.max_keep_alive    = 100;
    vtp.recondition_every = 16;
    vtp.keyframe_interval = keyframe_interval;
    auto tracker = sam3_create_visual_tracker(model, vtp);
    if (!tracker) { fprintf(stderr, "ERROR: tracker creation failed\n"); return result; }

    // Frame 0: encode + add instance
    if (!sam3_encode_image(*state, model, frames[0])) {
        fprintf(stderr, "ERROR: encode frame 0 failed\n"); return result;
    }
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;
    int inst_id = sam3_tracker_add_instance(*tracker, *state, model, pvs);
    if (inst_id < 0) {
        fprintf(stderr, "ERROR: add instance failed\n"); return result;
    }

    // Frames 1..N-1: propagate
    for (int f = 1; f < n_frames; f++) {
        auto r = sam3_propagate_frame(*tracker, *state, model, frames[f]);
        if (!r.detections.empty()) {
            const auto & det = r.detections[0];
            FrameMask & fm = result[f];
            fm.data = det.mask.data;
            fm.width = det.mask.width;
            fm.height = det.mask.height;
            fm.x0 = det.box.x0; fm.y0 = det.box.y0;
            fm.x1 = det.box.x1; fm.y1 = det.box.y1;
            fm.score = det.score;
            fm.valid = true;
        }
    }
    return result;
}

static std::vector<int> parse_k_values(const std::string & s) {
    std::vector<int> vals;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        int v = atoi(tok.c_str());
        if (v > 0) vals.push_back(v);
    }
    return vals;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string video_path = "data/test_video.mp4";
    float       px         = 315.0f;
    float       py         = 250.0f;
    int         n_frames   = 30;
    int         n_threads  = 4;
    bool        use_gpu    = true;
    int         encode_img_size = 0;
    std::string k_str      = "5,10,15,20,30";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"    && i+1 < argc) model_path = argv[++i];
        else if (arg == "--video"    && i+1 < argc) video_path = argv[++i];
        else if (arg == "--point-x"  && i+1 < argc) px = (float)atof(argv[++i]);
        else if (arg == "--point-y"  && i+1 < argc) py = (float)atof(argv[++i]);
        else if (arg == "--n-frames" && i+1 < argc) n_frames = atoi(argv[++i]);
        else if (arg == "--n-threads"&& i+1 < argc) n_threads = atoi(argv[++i]);
        else if (arg == "--no-gpu") use_gpu = false;
        else if (arg == "--k-values" && i+1 < argc) k_str = argv[++i];
        else if (arg == "--encode-img-size" && i+1 < argc) encode_img_size = atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s --model <path> [options]\n"
                "  --model <path>        Model .ggml file (required)\n"
                "  --video <path>        Video file (default: data/test_video.mp4)\n"
                "  --point-x <f>         Click X (default: 315.0)\n"
                "  --point-y <f>         Click Y (default: 250.0)\n"
                "  --n-frames <n>        Frames (default: 30)\n"
                "  --n-threads <n>       CPU threads (default: 4)\n"
                "  --no-gpu              CPU only\n"
                "  --k-values <list>     Comma-separated K values (default: 5,10,15,20,30)\n"
                "  --encode-img-size <n> Override encoder input resolution (default: model default)\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n");
        return 1;
    }

    auto k_values = parse_k_values(k_str);
    if (k_values.empty()) {
        fprintf(stderr, "ERROR: no valid K values\n");
        return 1;
    }

    // Decode all frames upfront
    fprintf(stderr, "Decoding %d frames from %s ...\n", n_frames, video_path.c_str());
    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; f++) {
        frames[f] = sam3_decode_video_frame(video_path, f);
        if (frames[f].data.empty()) {
            fprintf(stderr, "ERROR: failed to decode frame %d\n", f);
            return 1;
        }
    }

    // Load model
    sam3_params params;
    params.model_path     = model_path;
    params.use_gpu        = use_gpu;
    params.n_threads      = n_threads;
    params.encode_img_size = encode_img_size;

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "ERROR: load failed\n"); return 1; }

    // Run baseline (K=0, full pipeline every frame)
    fprintf(stderr, "\n=== Baseline (K=0, full pipeline every frame) ===\n");
    int64_t t0 = ggml_time_us();
    auto baseline = run_tracking(*model, frames, params, px, py, 0);
    double baseline_ms = (ggml_time_us() - t0) / 1000.0;
    fprintf(stderr, "Baseline total: %.0f ms (%.1f ms/frame)\n",
            baseline_ms, baseline_ms / std::max(1, n_frames - 1));

    // Run each K value and compare
    printf("\n");
    printf("============================================================================\n");
    printf("KEYFRAME QUALITY COMPARISON  —  %d frames, model=%s\n", n_frames, model_path.c_str());
    if (encode_img_size > 0)
        printf("encode_img_size=%d\n", encode_img_size);
    printf("============================================================================\n\n");
    printf("  %3s | %10s | %10s | %8s | %8s | %8s | %8s | %8s | %s\n",
           "K", "Time (ms)", "Speedup", "Mean IoU", "Min IoU", "P5 IoU",
           "Mean BIoU", "Min BIoU", "Keyframes");
    printf("------+------------+------------+----------+----------+----------+----------+----------+----------\n");

    for (int K : k_values) {
        fprintf(stderr, "\n=== K=%d ===\n", K);
        t0 = ggml_time_us();
        auto test = run_tracking(*model, frames, params, px, py, K);
        double test_ms = (ggml_time_us() - t0) / 1000.0;

        // Compute per-frame IoU
        std::vector<float> ious, bious;
        int n_valid = 0;
        for (int f = 1; f < n_frames; f++) {
            if (baseline[f].valid && test[f].valid) {
                ious.push_back(mask_iou(baseline[f], test[f]));
                bious.push_back(box_iou(baseline[f], test[f]));
                n_valid++;
            } else if (baseline[f].valid && !test[f].valid) {
                ious.push_back(0.0f);
                bious.push_back(0.0f);
                n_valid++;
            }
        }

        if (ious.empty()) {
            printf("  %3d | %10s | %10s | %8s | %8s | %8s | %8s | %8s | %s\n",
                   K, "-", "-", "-", "-", "-", "-", "-", "no data");
            continue;
        }

        std::sort(ious.begin(), ious.end());
        std::sort(bious.begin(), bious.end());

        float mean_iou = 0, mean_biou = 0;
        for (float v : ious) mean_iou += v;
        for (float v : bious) mean_biou += v;
        mean_iou /= ious.size();
        mean_biou /= bious.size();

        float min_iou = ious.front();
        float min_biou = bious.front();
        int p5_idx = (int)(0.05f * ious.size());
        float p5_iou = ious[p5_idx];

        int n_keyframes = 1; // frame 0 is always a keyframe
        for (int f = 1; f < n_frames; f++) {
            if ((f - 0) % K == 0) n_keyframes++; // approximate
        }

        double speedup = baseline_ms / std::max(1.0, test_ms);

        printf("  %3d | %10.0f | %9.1fx | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %d/%d\n",
               K, test_ms, speedup, mean_iou, min_iou, p5_iou,
               mean_biou, min_biou, n_keyframes, n_frames);

        // Per-frame detail to stderr
        fprintf(stderr, "  Per-frame IoU: ");
        for (int f = 1; f < n_frames && f <= 30; f++) {
            if (f - 1 < (int)ious.size()) {
                // Use unsorted order
                float iou_f = (baseline[f].valid && test[f].valid)
                    ? mask_iou(baseline[f], test[f]) : 0.0f;
                fprintf(stderr, "f%d=%.3f ", f, iou_f);
            }
        }
        fprintf(stderr, "\n");
    }

    printf("------+------------+------------+----------+----------+----------+----------+----------+----------\n");
    printf("\nBaseline (K=0): %.0f ms total, %.1f ms/frame\n\n",
           baseline_ms, baseline_ms / std::max(1, n_frames - 1));

    return 0;
}
