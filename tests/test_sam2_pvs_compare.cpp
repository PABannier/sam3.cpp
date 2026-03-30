// Test: Compare SAM2 PVS decoder output against Python reference.
// Set SAM2_DUMP_DIR to dump debug tensors.
// Usage: test_sam2_pvs_compare <model.ggml> <preprocessed.bin> <output_dir>
//   Point at center of image (600, 599 in original 1200x1198 image)

#include "sam3.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <sys/stat.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.ggml> <preprocessed.bin> <output_dir>\n", argv[0]);
        return 1;
    }

    mkdir(argv[3], 0755);

    // Load preprocessed image
    std::ifstream fin(argv[2], std::ios::binary);
    fin.seekg(0, std::ios::end);
    size_t sz = fin.tellg();
    fin.seekg(0);
    std::vector<float> img(sz / 4);
    fin.read(reinterpret_cast<char*>(img.data()), sz);

    // Load model
    sam3_params p;
    p.model_path = argv[1];
    p.n_threads = 8;
    p.use_gpu = false;
    auto model = sam3_load_model(p);
    if (!model) return 1;
    auto state = sam3_create_state(*model, p);
    if (!state) return 1;

    // Encode from preprocessed
    if (!sam3_encode_image_from_preprocessed(*state, *model, img.data(), 1024)) return 1;

    // Run PVS with single point at center
    // Original image is 1200x1198, point at (600, 599) = center
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({600.0f, 599.0f});
    pvs.multimask = true;

    auto result = sam3_segment_pvs(*state, *model, pvs);

    fprintf(stderr, "PVS result: %zu detections\n", result.detections.size());
    for (size_t i = 0; i < result.detections.size(); ++i) {
        fprintf(stderr, "  det %zu: iou=%.4f score=%.4f mask=%dx%d\n",
                i, result.detections[i].iou_score, result.detections[i].score,
                result.detections[i].mask.width, result.detections[i].mask.height);
    }

    // Dump outputs to output_dir
    if (result.detections.size() > 0) {
        // Save mask as PNG
        for (size_t i = 0; i < result.detections.size(); ++i) {
            char path[256];
            snprintf(path, sizeof(path), "%s/mask_%zu.png", argv[3], i);
            sam3_save_mask(result.detections[i].mask, path);
        }
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
