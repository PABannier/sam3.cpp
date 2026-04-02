/**
 * profile_edgetam — per-stage latency profiler for the EdgeTAM RepViT+FPN
 * image encoder.
 *
 * Loads an EdgeTAM model and a test image, then profiles the forward pass
 * broken into individually-timed sub-graphs (stem, stages 0-3, FPN neck).
 *
 * Usage:
 *   profile_edgetam [options]
 *
 * Options:
 *   --model <path>      EdgeTAM .ggml file (default: models/edgetam_f16.ggml)
 *   --image <path>      Test image          (default: data/test_image.jpg)
 *   --n-threads <n>     CPU threads          (default: 4)
 *   --n-warmup <n>      Warmup iterations    (default: 2)
 *   --n-iter <n>        Timed iterations     (default: 5)
 *   --cpu               Force CPU backend
 */

#include "sam3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    std::string model_path = "models/edgetam_f16.ggml";
    std::string image_path = "data/test_image.jpg";
    int n_threads = 4;
    int n_warmup  = 2;
    int n_iter    = 5;
    bool use_gpu  = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--n-threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-warmup") == 0 && i + 1 < argc) {
            n_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-iter") == 0 && i + 1 < argc) {
            n_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--model <path>] [--image <path>] "
                    "[--n-threads <n>] [--n-warmup <n>] [--n-iter <n>] [--cpu]\n", argv[0]);
            return 1;
        }
    }

    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    fprintf(stderr, "Test image:    %s\n", image_path.c_str());
    fprintf(stderr, "Backend:       %s\n", use_gpu ? "Metal (GPU)" : "CPU");
    fprintf(stderr, "\n");

    // Load model
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu    = use_gpu;
    params.n_threads  = n_threads;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    if (sam3_get_model_type(*model) != SAM3_MODEL_EDGETAM) {
        fprintf(stderr, "ERROR: model is not EdgeTAM (model_type=%d)\n",
                sam3_get_model_type(*model));
        return 1;
    }

    // Load image
    auto image = sam3_load_image(image_path);
    if (image.data.empty()) {
        fprintf(stderr, "ERROR: failed to load image '%s'\n", image_path.c_str());
        return 1;
    }
    fprintf(stderr, "Image: %dx%d (%d channels)\n\n", image.width, image.height, image.channels);

    // Profile
    bool ok = sam3_profile_edgetam_encode(*model, image, n_threads, n_warmup, n_iter);
    if (!ok) {
        fprintf(stderr, "ERROR: profiling failed\n");
        return 1;
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
