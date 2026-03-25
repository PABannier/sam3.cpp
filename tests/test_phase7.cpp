#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct phase7_case {
    std::string id;
    std::string label;
};

struct tensor_case {
    std::string name;
    std::string label;
};

struct report_row {
    std::string case_id;
    std::string label;
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float cosine = 0.0f;
    float tol = 1e-4f;
    std::string status;
    std::string note;
};

static std::vector<phase7_case> load_cases(const std::string & path) {
    std::vector<phase7_case> cases;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        size_t start = 0;
        while (start <= line.size()) {
            size_t end = line.find('\t', start);
            if (end == std::string::npos) {
                end = line.size();
            }
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
            if (end == line.size()) {
                break;
            }
        }
        while (fields.size() < 2) {
            fields.emplace_back();
        }
        cases.push_back({fields[0], fields[1]});
    }
    return cases;
}

static std::map<std::string, std::string> load_meta(const std::string & path) {
    std::map<std::string, std::string> kv;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

static std::vector<tensor_case> tensor_cases_for(const std::string & case_dir) {
    std::vector<tensor_case> cases;
    const auto meta = load_meta(case_dir + "/meta.txt");
    const int num_slots = meta.count("num_slots") ? std::stoi(meta.at("num_slots")) : 0;

    for (int i = 0; i < num_slots; ++i) {
        cases.push_back({"phase7_mem" + std::to_string(i) + "_pix_proj",
                         "Memory slot " + std::to_string(i) + " pixel projection"});
        cases.push_back({"phase7_mem" + std::to_string(i) + "_fused_input",
                         "Memory slot " + std::to_string(i) + " fused input"});
        cases.push_back({"phase7_mem" + std::to_string(i) + "_fuser0",
                         "Memory slot " + std::to_string(i) + " fuser block 0"});
        cases.push_back({"phase7_mem" + std::to_string(i) + "_fuser1",
                         "Memory slot " + std::to_string(i) + " fuser block 1"});
        cases.push_back({"phase7_mem" + std::to_string(i) + "_output",
                         "Memory slot " + std::to_string(i) + " encoder output"});
        cases.push_back({"phase7_obj_ptr" + std::to_string(i),
                         "Memory slot " + std::to_string(i) + " object pointer"});
    }

    cases.push_back({"phase7_mem_attn_input", "Memory attention input"});
    for (int i = 0; i < 4; ++i) {
        cases.push_back({"phase7_mem_attn_layer" + std::to_string(i) + "_after_sa",
                         "Memory attention layer " + std::to_string(i) + " after self-attn"});
        cases.push_back({"phase7_mem_attn_layer" + std::to_string(i) + "_after_ca",
                         "Memory attention layer " + std::to_string(i) + " after cross-attn"});
        cases.push_back({"phase7_mem_attn_layer" + std::to_string(i) + "_after_ffn",
                         "Memory attention layer " + std::to_string(i) + " after FFN"});
    }
    cases.push_back({"phase7_mem_attn_output", "Memory attention final output"});
    cases.push_back({"phase7_prop_masks", "Propagation mask logits"});
    cases.push_back({"phase7_prop_iou", "Propagation IoU"});
    cases.push_back({"phase7_prop_obj_score", "Propagation object score"});
    cases.push_back({"phase7_prop_sam_token", "Propagation SAM token"});
    return cases;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ref_dir> <model_path> <cases_tsv>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string model_path = argv[2];
    const std::string cases_path = argv[3];
    const std::string cpp_root = ref_dir + "/cpp_out_phase7";

    if (!ensure_dir(cpp_root)) {
        fprintf(stderr, "Failed to create %s\n", cpp_root.c_str());
        return 1;
    }

    auto cases = load_cases(cases_path);
    if (cases.empty()) {
        fprintf(stderr, "No phase 7 cases loaded from %s\n", cases_path.c_str());
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.n_threads = 1;
    params.use_gpu = false;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    bool overall_ok = true;
    std::vector<report_row> report;

    for (const auto & tc : cases) {
        fprintf(stderr, "\n=== Phase 7 Case %s ===\n", tc.id.c_str());

        const std::string case_ref_dir = ref_dir + "/" + tc.id;
        const std::string case_cpp_dir = cpp_root + "/" + tc.id;
        if (!ensure_dir(case_cpp_dir)) {
            fprintf(stderr, "Failed to create %s\n", case_cpp_dir.c_str());
            overall_ok = false;
            continue;
        }

        if (!sam3_test_dump_phase7_from_ref_inputs(*model, case_ref_dir, case_cpp_dir, params.n_threads)) {
            fprintf(stderr, "  [FAIL] could not dump C++ phase 7 tensors for %s\n", tc.id.c_str());
            overall_ok = false;
            continue;
        }

        for (const auto & tensor : tensor_cases_for(case_ref_dir)) {
            auto ref = load_ref_f32(case_ref_dir + "/" + tensor.name);
            auto got = load_ref_f32(case_cpp_dir + "/" + tensor.name);

            report_row row;
            row.case_id = tc.id;
            row.label = tensor.label;

            if (ref.data.empty() || got.data.empty()) {
                row.status = "FAIL";
                row.note = "missing reference or C++ output";
                overall_ok = false;
                report.push_back(std::move(row));
                fprintf(stderr, "  [FAIL] missing tensor %s for %s\n",
                        tensor.name.c_str(), tc.id.c_str());
                continue;
            }

            if (ref.numel() != got.numel()) {
                row.status = "FAIL";
                row.note = "numel mismatch";
                overall_ok = false;
                report.push_back(std::move(row));
                fprintf(stderr, "  [FAIL] numel mismatch for %s in %s\n",
                        tensor.name.c_str(), tc.id.c_str());
                continue;
            }

            auto r = compare_tensors(got.data.data(), ref.data.data(), ref.numel(), row.tol);
            const bool ok = r.max_diff <= row.tol;
            row.max_diff = r.max_diff;
            row.mean_diff = r.mean_diff;
            row.cosine = r.cosine_sim;
            row.status = ok ? "PASS" : "FAIL";
            if (ok) {
                row.note = "measured";
            } else {
                char note[160];
                snprintf(note, sizeof(note),
                         "worst_idx=%d got=%.6g ref=%.6g bad=%d/%d",
                         r.worst_index, r.worst_a, r.worst_b, r.n_bad, r.n_total);
                row.note = note;
                overall_ok = false;
                fprintf(stderr,
                        "  [FAIL] %-28s max=%.3e mean=%.3e cosine=%.8f idx=%d got=%.6g ref=%.6g\n",
                        tensor.name.c_str(), r.max_diff, r.mean_diff, r.cosine_sim,
                        r.worst_index, r.worst_a, r.worst_b);
            }

            report.push_back(std::move(row));
        }
    }

    fprintf(stderr, "\n## Numerical Precision Report\n\n");
    fprintf(stderr, "| Case | Operation | max_diff | mean_diff | cosine | Tolerance | Status | Notes |\n");
    fprintf(stderr, "|------|-----------|----------|-----------|--------|-----------|--------|-------|\n");
    for (const auto & row : report) {
        fprintf(stderr,
                "| %s | %s | %.3e | %.3e | %.8f | %.1e | %s | %s |\n",
                row.case_id.c_str(),
                row.label.c_str(),
                row.max_diff,
                row.mean_diff,
                row.cosine,
                row.tol,
                row.status.c_str(),
                row.note.c_str());
    }

    return overall_ok ? 0 : 1;
}
