#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sys/stat.h>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_path> <output_dir>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string output_dir = argv[2];

    // Create output directory
    SAM3_MKDIR(output_dir.c_str());

    // Load tokenizer from embedded data in model file
    if (!sam3_test_load_tokenizer(model_path)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", model_path.c_str());
        return 1;
    }

    // Tokenize "shoe"
    auto token_ids = sam3_test_tokenize("shoe");
    fprintf(stderr, "Token IDs: [");
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", token_ids[i]);
        if (token_ids[i] == 0 && i > 1) { fprintf(stderr, ", ..."); break; }
    }
    fprintf(stderr, "]\n");

    // Load model
    sam3_params params;
    params.model_path = model_path;
    params.n_threads = 1;
    params.use_gpu = false;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    // Dump text encoder tensors
    if (!sam3_test_dump_text_encoder(*model, token_ids, output_dir, params.n_threads)) {
        fprintf(stderr, "Failed to dump text encoder tensors\n");
        sam3_free_model(*model);
        return 1;
    }

    fprintf(stderr, "All tensors dumped to %s/\n", output_dir.c_str());
    sam3_free_model(*model);
    return 0;
}
