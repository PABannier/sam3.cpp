#include "sam3.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml>\n", argv[0]);
        return 1;
    }

    sam3_params params;
    params.model_path = argv[1];
    params.use_gpu    = true;
    params.n_threads  = 4;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    fprintf(stderr, "Model loaded successfully!\n");

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    fprintf(stderr, "State created successfully!\n");

    // Explicit cleanup before ggml global shutdown
    state.reset();
    sam3_free_model(*model);
    model.reset();

    fprintf(stderr, "Cleanup complete.\n");
    return 0;
}
