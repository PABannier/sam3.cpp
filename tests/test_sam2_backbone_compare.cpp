// Test: Compare SAM2 backbone + FPN outputs against Python.
// Set SAM2_DUMP_DIR env var to dump debug intermediates.
// Usage: SAM2_DUMP_DIR=/tmp/debug_sam2_cpp test_sam2_backbone_compare <model.ggml> <preprocessed.bin> <output_dir>

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

    // Dump FPN outputs via state tensor API
    for (int i = 0; i < 3; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        std::string path = std::string(argv[3]) + "/" + name;
        sam3_dump_state_tensor(*state, name, path);
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
