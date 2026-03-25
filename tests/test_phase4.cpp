#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct prompt_case {
    std::string id;
    std::string text;
};

struct summary_row {
    bool measured = false;
    bool ok = true;
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float cosine_sim = 1.0f;
    std::string worst_tensor;
    std::string note;
};

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::vector<prompt_case> load_prompt_cases(const std::string & path) {
    std::vector<prompt_case> cases;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }

        prompt_case pc;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            pc.id = "prompt_" + std::to_string(cases.size());
            pc.text = line;
        } else {
            pc.id = line.substr(0, tab);
            pc.text = line.substr(tab + 1);
        }
        cases.push_back(std::move(pc));
    }
    return cases;
}

static std::string category_for_tensor(const std::string & name) {
    if (name == "causal_mask")         return "causal_mask";
    if (name == "text_token_embed")    return "text_token_embed";
    if (name == "text_after_pos_embed") return "text_after_pos_embed";
    if (name == "text_final_ln")       return "text_final_ln";
    if (name == "text_features_2d")    return "text_features_2d";

    if (ends_with(name, "_after_ln1"))            return "block_after_ln1";
    if (ends_with(name, "_qkv"))                  return "block_qkv";
    if (ends_with(name, "_attn_out"))             return "block_attn_out";
    if (ends_with(name, "_after_attn_residual"))  return "block_after_attn_residual";
    if (ends_with(name, "_after_ln2"))            return "block_after_ln2";
    if (ends_with(name, "_mlp_fc1"))              return "block_mlp_fc1";
    if (ends_with(name, "_mlp_gelu"))             return "block_mlp_gelu";
    if (ends_with(name, "_mlp_out"))              return "block_mlp_out";
    if (ends_with(name, "_out"))                  return "block_out";

    return name;
}

static const char * label_for_category(const std::string & category) {
    if (category == "token_ids")                  return "Token IDs";
    if (category == "causal_mask")                return "Causal mask";
    if (category == "text_token_embed")           return "Token embedding";
    if (category == "text_after_pos_embed")       return "Embedding + position";
    if (category == "block_after_ln1")            return "Blocks after ln1";
    if (category == "block_qkv")                  return "Blocks qkv";
    if (category == "block_attn_out")             return "Blocks attn out";
    if (category == "block_after_attn_residual")  return "Blocks attn residual";
    if (category == "block_after_ln2")            return "Blocks after ln2";
    if (category == "block_mlp_fc1")              return "Blocks mlp fc1";
    if (category == "block_mlp_gelu")             return "Blocks mlp gelu";
    if (category == "block_mlp_out")              return "Blocks mlp out";
    if (category == "block_out")                  return "Blocks output";
    if (category == "text_final_ln")              return "Final layer norm";
    if (category == "text_features_2d")           return "Final text features";
    return category.c_str();
}

static std::vector<std::string> tensor_names() {
    std::vector<std::string> names = {
        "causal_mask",
        "text_token_embed",
        "text_after_pos_embed",
        "text_final_ln",
        "text_features_2d",
    };

    for (int i = 0; i < 24; ++i) {
        char name[64];

        snprintf(name, sizeof(name), "text_block_%02d_after_ln1", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_qkv", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_attn_out", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_attn_residual", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_after_ln2", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_fc1", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_gelu", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_mlp_out", i);
        names.emplace_back(name);

        snprintf(name, sizeof(name), "text_block_%02d_out", i);
        names.emplace_back(name);
    }

    return names;
}

static void update_summary(summary_row & row,
                           const std::string & tensor_name,
                           const compare_result & r,
                           bool ok) {
    if (!row.measured || r.max_diff >= row.max_diff) {
        row.max_diff = r.max_diff;
        row.mean_diff = r.mean_diff;
        row.cosine_sim = r.cosine_sim;
        row.worst_tensor = tensor_name;
    }
    row.measured = true;
    row.ok = row.ok && ok;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <ref_dir> <model_path> <tokenizer_dir> <prompts_tsv>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string model_path = argv[2];
    const std::string tokenizer_dir = argv[3];
    const std::string prompts_path = argv[4];
    const float atol = 1e-4f;

    auto prompts = load_prompt_cases(prompts_path);
    if (prompts.empty()) {
        fprintf(stderr, "No prompts loaded from %s\n", prompts_path.c_str());
        return 1;
    }

    if (!sam3_test_load_tokenizer(tokenizer_dir)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", tokenizer_dir.c_str());
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.tokenizer_dir = tokenizer_dir;
    params.n_threads = 1;
    params.use_gpu = false;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    const std::string cpp_root = ref_dir + "/cpp_out_phase4";
    if (!ensure_dir(cpp_root)) {
        fprintf(stderr, "Failed to create %s\n", cpp_root.c_str());
        return 1;
    }

    const std::vector<std::string> names = tensor_names();
    const std::vector<std::string> category_order = {
        "token_ids",
        "causal_mask",
        "text_token_embed",
        "text_after_pos_embed",
        "block_after_ln1",
        "block_qkv",
        "block_attn_out",
        "block_after_attn_residual",
        "block_after_ln2",
        "block_mlp_fc1",
        "block_mlp_gelu",
        "block_mlp_out",
        "block_out",
        "text_final_ln",
        "text_features_2d",
    };

    std::map<std::string, std::map<std::string, summary_row>> report;
    bool overall_ok = true;

    for (const auto & prompt : prompts) {
        fprintf(stderr, "\n=== Prompt %s: \"%s\" ===\n", prompt.id.c_str(), prompt.text.c_str());

        const std::string prompt_ref_dir = ref_dir + "/" + prompt.id;
        const std::string prompt_cpp_dir = cpp_root + "/" + prompt.id;
        if (!ensure_dir(prompt_cpp_dir)) {
            fprintf(stderr, "Failed to create %s\n", prompt_cpp_dir.c_str());
            overall_ok = false;
            continue;
        }

        auto ref_token_ids = load_ref_i32(prompt_ref_dir + "/token_ids");
        if (ref_token_ids.data.empty()) {
            fprintf(stderr, "Missing reference token_ids for %s\n", prompt.id.c_str());
            overall_ok = false;
            continue;
        }

        auto cpp_token_ids = sam3_test_tokenize(prompt.text);
        int token_bad = compare_exact_i32(cpp_token_ids, ref_token_ids.data);
        summary_row token_row;
        token_row.measured = true;
        token_row.ok = (token_bad == 0);
        token_row.max_diff = token_bad == 0 ? 0.0f : 1.0f;
        token_row.mean_diff = token_bad == 0 ? 0.0f : 1.0f;
        token_row.cosine_sim = token_bad == 0 ? 1.0f : 0.0f;
        token_row.note = token_bad == 0 ? "exact int compare" : "tokenizer mismatch";
        report[prompt.id]["token_ids"] = token_row;
        if (token_bad != 0) {
            fprintf(stderr, "  [FAIL] token_ids mismatch for %s\n", prompt.id.c_str());
            overall_ok = false;
        }

        if (!sam3_test_dump_text_encoder(*model, ref_token_ids.data, prompt_cpp_dir, params.n_threads)) {
            fprintf(stderr, "  [FAIL] could not dump C++ text encoder tensors for %s\n", prompt.id.c_str());
            overall_ok = false;
            continue;
        }

        for (const auto & name : names) {
            auto ref = load_ref_f32(prompt_ref_dir + "/" + name);
            auto got = load_ref_f32(prompt_cpp_dir + "/" + name);
            const std::string category = category_for_tensor(name);

            if (ref.data.empty() || got.data.empty()) {
                auto & row = report[prompt.id][category];
                row.ok = false;
                row.note = "missing reference or C++ output";
                overall_ok = false;
                fprintf(stderr, "  [FAIL] missing tensor %s for %s\n", name.c_str(), prompt.id.c_str());
                continue;
            }

            if (ref.numel() != got.numel()) {
                auto & row = report[prompt.id][category];
                row.ok = false;
                row.note = "shape mismatch";
                overall_ok = false;
                fprintf(stderr, "  [FAIL] shape mismatch for %s (%d vs %d)\n",
                        name.c_str(), got.numel(), ref.numel());
                continue;
            }

            auto r = compare_tensors(got.data.data(), ref.data.data(), ref.numel(), atol);
            bool ok = r.max_diff <= atol;
            update_summary(report[prompt.id][category], name, r, ok);
            if (!ok) {
                overall_ok = false;
                fprintf(stderr,
                        "  [FAIL] %-32s max=%.3e mean=%.3e cosine=%.8f\n",
                        name.c_str(), r.max_diff, r.mean_diff, r.cosine_sim);
            }
        }

        for (const auto & category : category_order) {
            auto it = report[prompt.id].find(category);
            if (it == report[prompt.id].end() || !it->second.measured) {
                if (category == "token_ids") {
                    continue;
                }
                fprintf(stderr, "  [FAIL] no measurements recorded for %s/%s\n",
                        prompt.id.c_str(), category.c_str());
                overall_ok = false;
                report[prompt.id][category].ok = false;
                report[prompt.id][category].note = "untested";
            }
        }
    }

    fprintf(stderr, "\n## Numerical Precision Report\n\n");
    fprintf(stderr, "| Prompt | Operation | max_diff | mean_diff | cosine | Tolerance | Status | Notes |\n");
    fprintf(stderr, "|--------|-----------|----------|-----------|--------|-----------|--------|-------|\n");
    for (const auto & prompt : prompts) {
        for (const auto & category : category_order) {
            const auto it = report[prompt.id].find(category);
            summary_row row;
            if (it != report[prompt.id].end()) {
                row = it->second;
            }

            const char * status = !row.measured && category != "token_ids"
                ? "UNTESTED"
                : (row.ok ? "PASS" : "FAIL");

            std::string note = row.note;
            if (note.empty() && !row.worst_tensor.empty()) {
                note = "worst=" + row.worst_tensor;
            }

            fprintf(stderr,
                    "| %s | %s | %.3e | %.3e | %.8f | %.1e | %s | %s |\n",
                    prompt.id.c_str(),
                    label_for_category(category),
                    row.max_diff,
                    row.mean_diff,
                    row.cosine_sim,
                    atol,
                    status,
                    note.empty() ? "measured" : note.c_str());
        }
    }

    return overall_ok ? 0 : 1;
}
