#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct phase6_case {
    std::string id;
    sam3_pvs_params params;
};

struct tensor_case {
    std::string name;
    std::string label;
};

struct report_row {
    bool measured = false;
    bool ok = false;
    compare_result r;
    std::string note;
};

static std::vector<sam3_point> parse_points(const std::string & field) {
    std::vector<sam3_point> pts;
    if (field.empty()) {
        return pts;
    }

    size_t start = 0;
    while (start < field.size()) {
        size_t end = field.find('|', start);
        if (end == std::string::npos) {
            end = field.size();
        }

        const std::string part = field.substr(start, end - start);
        size_t colon = part.find(':');
        if (colon != std::string::npos) {
            sam3_point pt;
            pt.x = std::stof(part.substr(0, colon));
            pt.y = std::stof(part.substr(colon + 1));
            pts.push_back(pt);
        }

        start = end + 1;
    }

    return pts;
}

static bool parse_box(const std::string & field, sam3_box & box) {
    if (field.empty()) {
        return false;
    }

    float vals[4];
    size_t start = 0;
    for (int i = 0; i < 4; ++i) {
        size_t end = field.find(':', start);
        if (end == std::string::npos) {
            end = field.size();
        }
        vals[i] = std::stof(field.substr(start, end - start));
        start = end + 1;
    }

    box = {vals[0], vals[1], vals[2], vals[3]};
    return true;
}

static std::vector<phase6_case> load_cases(const std::string & path) {
    std::vector<phase6_case> cases;
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
        while (fields.size() < 5) {
            fields.emplace_back();
        }

        phase6_case tc;
        tc.id = fields[0];
        tc.params.multimask = fields[1] == "1";
        tc.params.pos_points = parse_points(fields[2]);
        tc.params.neg_points = parse_points(fields[3]);
        tc.params.use_box = parse_box(fields[4], tc.params.box);
        cases.push_back(std::move(tc));
    }
    return cases;
}

static std::vector<tensor_case> tensor_cases() {
    return {
        {"sam_pe_sparse", "Prompt sparse embeddings"},
        {"sam_pe_dense", "Prompt dense embeddings"},
        {"sam_pe_image_pe", "Prompt image PE"},
        {"sam_dec_image_feats", "Decoder image features"},
        {"sam_dec_tokens_initial", "Decoder initial tokens"},
        {"sam_dec_block0_queries", "Decoder block 0 queries"},
        {"sam_dec_block0_keys", "Decoder block 0 keys"},
        {"sam_dec_block1_queries", "Decoder block 1 queries"},
        {"sam_dec_block1_keys", "Decoder block 1 keys"},
        {"sam_dec_final_queries", "Decoder final queries"},
        {"sam_dec_feat_s1_proj", "Decoder s1 projection"},
        {"sam_dec_feat_s0_proj", "Decoder s0 projection"},
        {"sam_dec_upscaled", "Decoder upscaled embedding"},
        {"sam_dec_mask_tokens", "Decoder mask tokens"},
        {"sam_dec_masks", "Decoder mask logits"},
        {"sam_dec_iou", "Decoder IoU predictions"},
        {"sam_dec_obj_score", "Decoder object score logit"},
        {"sam_dec_sam_token", "Decoder SAM token"},
    };
}

static float tolerance_for(const std::string & name) {
    if (name == "sam_dec_block1_queries" ||
        name == "sam_dec_final_queries") {
        return 2e-4f;
    }
    if (name == "sam_dec_upscaled") {
        return 8e-4f;
    }
    if (name == "sam_dec_masks") {
        return 8e-3f;
    }
    return 1e-4f;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <ref_dir> <prephase_ref_dir> <model_path> <cases_tsv>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string prephase_ref_dir = argv[2];
    const std::string model_path = argv[3];
    const std::string cases_path = argv[4];
    const std::string cpp_root = ref_dir + "/cpp_out_phase6";

    auto cases = load_cases(cases_path);
    if (cases.empty()) {
        fprintf(stderr, "No phase 6 cases loaded from %s\n", cases_path.c_str());
        return 1;
    }

    if (!ensure_dir(cpp_root)) {
        fprintf(stderr, "Failed to create %s\n", cpp_root.c_str());
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

    const auto tensors = tensor_cases();
    bool overall_ok = true;

    struct row_print {
        std::string case_id;
        std::string label;
        float max_diff = 0.0f;
        float mean_diff = 0.0f;
        float cosine = 0.0f;
        float tol = 0.0f;
        std::string status;
        std::string note;
    };
    std::vector<row_print> report;

    for (const auto & tc : cases) {
        fprintf(stderr, "\n=== Phase 6 Case %s ===\n", tc.id.c_str());

        const std::string case_ref_dir = ref_dir + "/" + tc.id;
        const std::string case_cpp_dir = cpp_root + "/" + tc.id;
        if (!ensure_dir(case_cpp_dir)) {
            fprintf(stderr, "Failed to create %s\n", case_cpp_dir.c_str());
            overall_ok = false;
            continue;
        }

        if (!sam3_test_dump_phase6_from_ref_inputs(*model, prephase_ref_dir, tc.params,
                                                   case_cpp_dir, params.n_threads)) {
            fprintf(stderr, "  [FAIL] could not dump C++ phase 6 tensors for %s\n", tc.id.c_str());
            overall_ok = false;
            continue;
        }

        for (const auto & tensor : tensors) {
            auto ref = load_ref_f32(case_ref_dir + "/" + tensor.name);
            auto got = load_ref_f32(case_cpp_dir + "/" + tensor.name);

            row_print row;
            row.case_id = tc.id;
            row.label = tensor.label;
            row.tol = tolerance_for(tensor.name);

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
                        "  [FAIL] %-24s max=%.3e mean=%.3e cosine=%.8f idx=%d got=%.6g ref=%.6g\n",
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
