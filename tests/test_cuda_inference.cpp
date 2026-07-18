/**
 * Simple test to validate that CUDA works during inference
 */

#include "sam3.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    printf("=== CUDA inference test for sam3.cpp ===\n\n");

    // Configuration
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = true;
    params.n_threads = 4;

    printf("1. Loading model...\n");
    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }
    printf("   ✓ Model loaded\n\n");

    // Create state
    printf("2. Creating inference state...\n");
    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Error: failed to create state\n");
        return 1;
    }
    printf("   ✓ State created\n\n");

    // Load image
    printf("3. Loading image: %s\n", image_path);
    auto image = sam3_load_image(image_path);
    if (image.data.empty()) {
        fprintf(stderr, "Error: failed to load image\n");
        return 1;
    }
    printf("   ✓ Image loaded: %dx%d\n\n", image.width, image.height);

    // Encode image (uses GPU if CUDA is active)
    printf("4. Encoding image (uses CUDA if available)...\n");
    bool success = sam3_encode_image(*state, *model, image);
    if (!success) {
        fprintf(stderr, "Error: encoding failed\n");
        return 1;
    }
    printf("   ✓ Encoding successful\n\n");

    // Segmentation with a point
    printf("5. Testing segmentation with a point...\n");
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({image.width / 2.0f, image.height / 2.0f});

    auto result = sam3_segment_pvs(*state, *model, pvs);
    if (result.detections.empty()) {
        fprintf(stderr, "Warning: no detections\n");
    } else {
        printf("   ✓ Segmentation successful: %zu detection(s)\n", result.detections.size());
        for (size_t i = 0; i < result.detections.size(); ++i) {
            printf("     - Detection %zu: IoU=%.3f, size=%dx%d\n",
                   i, result.detections[i].iou_score,
                   result.detections[i].mask.width,
                   result.detections[i].mask.height);
        }
    }

    printf("\n=== ✓ CUDA inference test passed! ===\n");
    printf("\nIf you saw 'using CUDA backend' at the beginning,\n");
    printf("then CUDA is working correctly for inference.\n");

    return 0;
}

