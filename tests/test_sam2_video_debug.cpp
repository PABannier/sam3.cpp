/**
 * test_sam2_video_debug — Debug SAM2 video tracking pipeline.
 *
 * Loads bedroom video frames, runs video tracking for 5 frames,
 * and dumps intermediate tensors for comparison with Python reference.
 *
 * Usage:
 *   test_sam2_video_debug <model.ggml> <video_frames_dir> [point_x point_y]
 *
 * Example:
 *   test_sam2_video_debug models/sam2.1_hiera_tiny_f32.ggml \
 *       ~/Documents/sam2/notebooks/videos/bedroom 210 350
 */
#include "sam3.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

static void dump_tensor_f32(const char* dir, const char* name,
                            const float* data, int n,
                            const char* shape_str) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to write %s\n", path); return; }
    fwrite(data, sizeof(float), n, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/%s.shape", dir, name);
    f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%s", shape_str);
    fclose(f);

    // Print stats
    float mn = data[0], mx = data[0];
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  [DUMP] %-50s shape=%-20s range=[%.4f, %.4f] mean=%.6f\n",
            name, shape_str, mn, mx, sum / n);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <video_frames_dir> [point_x point_y]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* frames_dir = argv[2];
    float point_x = (argc > 3) ? atof(argv[3]) : 210.0f;
    float point_y = (argc > 4) ? atof(argv[4]) : 350.0f;

    const char* dump_dir = "/tmp/debug_sam2_cpp";
    mkdir(dump_dir, 0755);

    const int N_FRAMES = argc > 5 ? atoi(argv[5]) : 5;
    int encode_img_size = (argc > 6) ? atoi(argv[6]) : 0;

    // Load model
    fprintf(stderr, "Loading model: %s\n", model_path);
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;  // CPU for reproducibility
    params.encode_img_size = encode_img_size;
    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Create state
    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    // Load frames as JPEG images
    std::vector<sam3_image> frames;
    for (int i = 0; i < N_FRAMES + 1; ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%05d.jpg", frames_dir, i);
        sam3_image img = sam3_load_image(path);
        if (img.data.empty()) {
            fprintf(stderr, "Failed to load frame %d: %s\n", i, path);
            return 1;
        }
        fprintf(stderr, "Loaded frame %d: %dx%d\n", i, img.width, img.height);
        frames.push_back(std::move(img));
    }

    // Create visual tracker
    sam3_visual_track_params vtp;
    vtp.max_keep_alive = 100;
    vtp.recondition_every = 16;
    auto tracker = sam3_create_visual_tracker(*model, vtp);
    if (!tracker) {
        fprintf(stderr, "Failed to create tracker\n");
        return 1;
    }

    // ── Frame 0: Encode + add instance with point prompt ──────────────────
    fprintf(stderr, "\n=== Frame 0: Encode + Add Instance ===\n");

    // Encode frame 0
    if (!sam3_encode_image(*state, *model, frames[0])) {
        fprintf(stderr, "Failed to encode frame 0\n");
        return 1;
    }

    // Dump neck features for frame 0
    // state.neck_trk[0,1,2] are the FPN features
    sam3_dump_state_tensor(*state, "neck_trk_0", std::string(dump_dir) + "/f0_neck_trk_0");
    sam3_dump_state_tensor(*state, "neck_trk_1", std::string(dump_dir) + "/f0_neck_trk_1");
    sam3_dump_state_tensor(*state, "neck_trk_2", std::string(dump_dir) + "/f0_neck_trk_2");

    // Add instance with point prompt
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({point_x, point_y});
    pvs.multimask = false;

    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    fprintf(stderr, "Added instance: %d\n", inst_id);
    if (inst_id < 0) {
        fprintf(stderr, "Failed to add instance\n");
        return 1;
    }

    // ── Propagate frames 1-4 ──────────────────────────────────────────────
    for (int fi = 1; fi <= N_FRAMES; ++fi) {
        fprintf(stderr, "\n=== Frame %d: Propagate ===\n", fi);

        auto result = sam3_propagate_frame(*tracker, *state, *model, frames[fi]);

        fprintf(stderr, "  Detections: %zu\n", result.detections.size());
        for (size_t d = 0; d < result.detections.size(); ++d) {
            const auto& det = result.detections[d];
            int fg = 0;
            for (auto v : det.mask.data) if (v > 127) fg++;
            float pct = 100.0f * fg / std::max(1, (int)det.mask.data.size());
            fprintf(stderr, "  det[%zu] inst=%d score=%.4f fg=%d (%.1f%%) mask=%dx%d\n",
                    d, det.instance_id, det.score, fg, pct,
                    det.mask.width, det.mask.height);
        }

        // Dump neck features
        char buf[128];
        snprintf(buf, sizeof(buf), "f%d_neck_trk_2", fi);
        sam3_dump_state_tensor(*state, "neck_trk_2",
                               std::string(dump_dir) + "/" + buf);

        // Dump output mask
        if (!result.detections.empty()) {
            const auto& det = result.detections[0];
            std::vector<float> mask_f(det.mask.data.size());
            for (size_t i = 0; i < det.mask.data.size(); ++i)
                mask_f[i] = det.mask.data[i] > 127 ? 1.0f : 0.0f;
            snprintf(buf, sizeof(buf), "f%d_output_mask_binary", fi);
            char shape[64];
            snprintf(shape, sizeof(shape), "%d,%d", det.mask.height, det.mask.width);
            dump_tensor_f32(dump_dir, buf, mask_f.data(), (int)mask_f.size(), shape);
        }
    }

    fprintf(stderr, "\nDone. Tensors dumped to %s/\n", dump_dir);
    return 0;
}
