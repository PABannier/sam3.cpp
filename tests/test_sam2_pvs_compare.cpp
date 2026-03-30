// Test: Compare SAM2 PVS decoder output against Python reference.
// Set SAM2_DUMP_DIR to dump debug tensors.
// Usage: test_sam2_pvs_compare <model.ggml> <preprocessed.bin> <output_dir> [orig_w orig_h point_x point_y]
//   Default: orig 1200x1198, point at (600, 599)

#include "sam3.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <sys/stat.h>

// Hack: set orig dims on state (state is opaque, but we know its layout)
// This is needed because encode_image_from_preprocessed sets orig to img_size
extern void sam3_state_set_orig_dims(sam3_state& state, int w, int h);

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.ggml> <preprocessed.bin> <output_dir> [orig_w orig_h point_x point_y]\n", argv[0]);
        return 1;
    }

    int orig_w = (argc > 4) ? atoi(argv[4]) : 1200;
    int orig_h = (argc > 5) ? atoi(argv[5]) : 1198;
    float point_x = (argc > 6) ? atof(argv[6]) : 600.0f;
    float point_y = (argc > 7) ? atof(argv[7]) : 599.0f;

    mkdir(argv[3], 0755);

    std::ifstream fin(argv[2], std::ios::binary);
    fin.seekg(0, std::ios::end);
    size_t sz = fin.tellg();
    fin.seekg(0);
    std::vector<float> img(sz / 4);
    fin.read(reinterpret_cast<char*>(img.data()), sz);

    sam3_params p;
    p.model_path = argv[1];
    p.n_threads = 8;
    p.use_gpu = false;
    auto model = sam3_load_model(p);
    if (!model) return 1;
    auto state = sam3_create_state(*model, p);
    if (!state) return 1;

    if (!sam3_encode_image_from_preprocessed(*state, *model, img.data(), 1024)) return 1;

    // Override original image dimensions (encode_from_preprocessed sets them to 1024)
    sam3_state_set_orig_dims(*state, orig_w, orig_h);

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({point_x, point_y});
    pvs.multimask = true;

    auto result = sam3_segment_pvs(*state, *model, pvs);

    fprintf(stderr, "PVS result: %zu detections\n", result.detections.size());
    for (size_t i = 0; i < result.detections.size(); ++i) {
        fprintf(stderr, "  det %zu: iou=%.4f score=%.4f mask=%dx%d\n",
                i, result.detections[i].iou_score, result.detections[i].score,
                result.detections[i].mask.width, result.detections[i].mask.height);
    }

    if (result.detections.size() > 0) {
        for (size_t i = 0; i < result.detections.size(); ++i) {
            char path[256];
            snprintf(path, sizeof(path), "%s/mask_%zu.png", argv[3], i);
            sam3_save_mask(result.detections[i].mask, path);
        }
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
