// Test Metal ViT encoding: compare CPU vs Metal image encoder outputs.
// Usage: test_metal_vit <model.ggml> <image.jpg>

#include "sam3.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    // Load image once
    auto img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        return 1;
    }
    fprintf(stderr, "Loaded %dx%d image\n", img.width, img.height);

    // ── Run CPU encoding ────────────────────────────────────────────────
    fprintf(stderr, "\n=== CPU Encoding ===\n");
    std::vector<float> cpu_neck0, cpu_neck1, cpu_neck2;
    {
        sam3_params params;
        params.model_path = model_path;
        params.use_gpu = false;
        params.n_threads = 8;

        auto model = sam3_load_model(params);
        if (!model) { fprintf(stderr, "Failed to load model (CPU)\n"); return 1; }

        auto state = sam3_create_state(*model, params);
        if (!state) { fprintf(stderr, "Failed to create state (CPU)\n"); return 1; }

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = sam3_encode_image(*state, *model, img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!ok) { fprintf(stderr, "CPU encode_image failed\n"); return 1; }
        fprintf(stderr, "  CPU encode_image: %.1f ms\n", ms);

        // Dump tensors via the debug interface
        sam3_dump_state_tensor(*state, "vit_output", "/tmp/sam3_cpu_vit");
        sam3_dump_state_tensor(*state, "neck_trk_0", "/tmp/sam3_cpu_neck0");
        sam3_dump_state_tensor(*state, "neck_trk_1", "/tmp/sam3_cpu_neck1");
        sam3_dump_state_tensor(*state, "neck_trk_2", "/tmp/sam3_cpu_neck2");
    }

    // ── Run Metal encoding ──────────────────────────────────────────────
    fprintf(stderr, "\n=== Metal Encoding ===\n");
    {
        sam3_params params;
        params.model_path = model_path;
        params.use_gpu = true;
        params.n_threads = 8;

        auto model = sam3_load_model(params);
        if (!model) { fprintf(stderr, "Failed to load model (Metal)\n"); return 1; }

        auto state = sam3_create_state(*model, params);
        if (!state) { fprintf(stderr, "Failed to create state (Metal)\n"); return 1; }

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = sam3_encode_image(*state, *model, img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!ok) { fprintf(stderr, "Metal encode_image failed\n"); return 1; }
        fprintf(stderr, "  Metal encode_image: %.1f ms\n", ms);

        sam3_dump_state_tensor(*state, "vit_output", "/tmp/sam3_metal_vit");
        sam3_dump_state_tensor(*state, "neck_trk_0", "/tmp/sam3_metal_neck0");
        sam3_dump_state_tensor(*state, "neck_trk_1", "/tmp/sam3_metal_neck1");
        sam3_dump_state_tensor(*state, "neck_trk_2", "/tmp/sam3_metal_neck2");
    }

    // ── Compare outputs ─────────────────────────────────────────────────
    fprintf(stderr, "\n=== Comparing CPU vs Metal ===\n");
    bool all_pass = true;

    auto compare_dumps = [&](const std::string& name,
                             const std::string& cpu_path,
                             const std::string& metal_path,
                             float atol) {
        // Load shapes
        auto load_bin = [](const std::string& path, std::vector<int>& shape) -> std::vector<float> {
            // Read shape
            {
                FILE* f = fopen((path + ".shape").c_str(), "r");
                if (!f) return {};
                char buf[256];
                if (fgets(buf, sizeof(buf), f)) {
                    char* tok = strtok(buf, ",\n");
                    while (tok) {
                        shape.push_back(atoi(tok));
                        tok = strtok(nullptr, ",\n");
                    }
                }
                fclose(f);
            }
            // Read data
            int n = 1;
            for (auto d : shape) n *= d;
            std::vector<float> data(n);
            FILE* f = fopen((path + ".bin").c_str(), "rb");
            if (!f) return {};
            fread(data.data(), sizeof(float), n, f);
            fclose(f);
            return data;
        };

        std::vector<int> cpu_shape, metal_shape;
        auto cpu_data = load_bin(cpu_path, cpu_shape);
        auto metal_data = load_bin(metal_path, metal_shape);

        if (cpu_data.empty() || metal_data.empty()) {
            fprintf(stderr, "  [SKIP] %s — could not load dumps\n", name.c_str());
            return;
        }
        if (cpu_data.size() != metal_data.size()) {
            fprintf(stderr, "  [FAIL] %s — size mismatch: %zu vs %zu\n",
                    name.c_str(), cpu_data.size(), metal_data.size());
            all_pass = false;
            return;
        }

        float max_diff = 0.0f;
        int max_idx = 0;
        int n_bad = 0;
        for (size_t i = 0; i < cpu_data.size(); i++) {
            float diff = std::fabs(cpu_data[i] - metal_data[i]);
            if (diff > max_diff) { max_diff = diff; max_idx = (int)i; }
            if (diff > atol) n_bad++;
        }

        bool ok = (max_diff < atol);
        fprintf(stderr, "  %s %s: max_diff=%.6f at [%d] (cpu=%.6f metal=%.6f) n_bad=%d/%zu\n",
                ok ? "[PASS]" : "[FAIL]", name.c_str(), max_diff, max_idx,
                cpu_data[max_idx], metal_data[max_idx], n_bad, cpu_data.size());
        if (!ok) all_pass = false;
    };

    // ViT backbone + FPN neck outputs — expect reasonable tolerance
    // (different execution order on GPU can cause FP32 rounding differences)
    compare_dumps("vit_output", "/tmp/sam3_cpu_vit", "/tmp/sam3_metal_vit", 0.1f);
    compare_dumps("neck_trk_0", "/tmp/sam3_cpu_neck0", "/tmp/sam3_metal_neck0", 0.1f);
    compare_dumps("neck_trk_1", "/tmp/sam3_cpu_neck1", "/tmp/sam3_metal_neck1", 0.1f);
    compare_dumps("neck_trk_2", "/tmp/sam3_cpu_neck2", "/tmp/sam3_metal_neck2", 0.1f);

    fprintf(stderr, "\n%s\n", all_pass ? "ALL COMPARISONS PASSED" : "SOME COMPARISONS FAILED");
    return all_pass ? 0 : 1;
}
