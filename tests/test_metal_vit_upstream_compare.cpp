#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "sam3.h"
#include "test_utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct tensor_spec {
    std::string label;
    std::string name;
};

struct backend_run_result {
    double elapsed_ms = 0.0;
    std::map<std::string, sam3_tensor_info> infos;
};

struct isolated_run_result {
    double elapsed_ms = 0.0;
    std::vector<float> output;
    std::vector<int> shape;
};

static std::string format_shape(const std::vector<int> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static std::string make_block_name(int block_idx, const char * suffix) {
    char name[64];
    snprintf(name, sizeof(name), "dbg_block_%d_%s", block_idx, suffix);
    return name;
}

static bool dump_named_tensors(const sam3_state & state,
                               const std::vector<std::string> & names,
                               const std::string & output_dir,
                               std::map<std::string, sam3_tensor_info> & infos) {
    for (const auto & name : names) {
        sam3_tensor_info info;
        if (!sam3_get_state_tensor_info(state, name, info)) {
            fprintf(stderr, "missing tensor info for %s\n", name.c_str());
            return false;
        }
        infos[name] = info;
        if (!sam3_dump_state_tensor(state, name, output_dir + "/" + name)) {
            fprintf(stderr, "failed to dump %s\n", name.c_str());
            return false;
        }
    }
    return true;
}

static ggml_backend_t create_backend(bool use_gpu, int n_threads) {
#ifdef GGML_USE_METAL
    if (use_gpu) {
        return ggml_backend_metal_init();
    }
#else
    (void) use_gpu;
#endif
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    return backend;
}

static std::vector<int> normalize_shape_4d(const std::vector<int> & shape) {
    std::vector<int> out = shape;
    while (out.size() < 4) {
        out.push_back(1);
    }
    return out;
}

static ggml_tensor * new_tensor_4d_from_shape(ggml_context * ctx,
                                              ggml_type type,
                                              const std::vector<int> & shape) {
    const std::vector<int> s = normalize_shape_4d(shape);
    return ggml_new_tensor_4d(ctx, type, s[0], s[1], s[2], s[3]);
}

static bool load_block_norm_params(const std::string & model_path,
                                   int block_idx,
                                   ref_tensor_f32 & weight,
                                   ref_tensor_f32 & bias) {
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 1;

    auto model = sam3_load_model(params);
    if (!model) {
        return false;
    }

    const std::string tmp_dir = "/tmp/sam3_vit_upstream_model";
    ensure_dir(tmp_dir);
    const std::string prefix = "vit.blocks." + std::to_string(block_idx) + ".norm1";
    if (!sam3_dump_model_tensor(*model, prefix + ".weight", tmp_dir + "/norm1_weight")) {
        return false;
    }
    if (!sam3_dump_model_tensor(*model, prefix + ".bias", tmp_dir + "/norm1_bias")) {
        return false;
    }

    weight = load_ref_f32(tmp_dir + "/norm1_weight");
    bias = load_ref_f32(tmp_dir + "/norm1_bias");
    sam3_free_model(*model);
    model.reset();
    return !weight.data.empty() && !bias.data.empty();
}

static isolated_run_result run_isolated_norm(ggml_backend_t backend,
                                             const ref_tensor_f32 & x_in,
                                             const ref_tensor_f32 & weight,
                                             const ref_tensor_f32 & bias) {
    isolated_run_result result;

    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    ggml_tensor * x = new_tensor_4d_from_shape(ctx, GGML_TYPE_F32, x_in.shape);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, weight.shape[0]);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bias.shape[0]);
    ggml_set_input(x);
    ggml_set_input(w);
    ggml_set_input(b);

    ggml_tensor * y = ggml_norm(ctx, x, 1e-5f);
    y = ggml_mul_inplace(ctx, y, w);
    y = ggml_add_inplace(ctx, y, b);
    ggml_set_output(y);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(x, x_in.data.data(), 0, x_in.data.size() * sizeof(float));
    ggml_backend_tensor_set(w, weight.data.data(), 0, weight.data.size() * sizeof(float));
    ggml_backend_tensor_set(b, bias.data.data(), 0, bias.data.size() * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (status == GGML_STATUS_SUCCESS) {
        result.shape = {
            (int) y->ne[0],
            (int) y->ne[1],
            (int) y->ne[2],
            (int) y->ne[3],
        };
        result.output.resize((size_t) ggml_nelements(y));
        ggml_backend_tensor_get(y, result.output.data(), 0, result.output.size() * sizeof(float));
    } else {
        fprintf(stderr, "isolated norm failed: %s\n", ggml_status_to_string(status));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static bool run_backend(const std::string & model_path,
                        const std::vector<float> & chw_data,
                        int img_size,
                        bool use_gpu,
                        int n_threads,
                        int debug_block_idx,
                        const std::vector<std::string> & names,
                        const std::string & dump_dir,
                        backend_run_result & result) {
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = use_gpu;
    params.n_threads = n_threads;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "failed to load model for %s\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "failed to create state for %s\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    const bool ok = sam3_encode_vit_from_preprocessed_selective(
            *state, *model, chw_data.data(), img_size, names);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
        fprintf(stderr, "%s selective ViT encode failed\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    if (!dump_named_tensors(*state, names, dump_dir, result.infos)) {
        return false;
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();
    return true;
}

static std::vector<tensor_spec> default_specs_for_block(int block_idx) {
    std::vector<tensor_spec> specs;

    if (block_idx <= 0) {
        specs.push_back({"block_0_input / norm1_input", "dbg_after_ln_pre"});
    } else {
        specs.push_back({"block_input / norm1_input", make_block_name(block_idx - 1, "out")});
    }

    specs.push_back({"norm1_output / qkv_input", make_block_name(block_idx, "norm1")});
    specs.push_back({"qkv_proj_output", make_block_name(block_idx, "qkv_proj")});
    specs.push_back({"attn_output", make_block_name(block_idx, "attn_out")});
    specs.push_back({"attn_proj_output", make_block_name(block_idx, "attn_proj")});
    specs.push_back({"resid1_output", make_block_name(block_idx, "resid1")});
    specs.push_back({"mlp_output", make_block_name(block_idx, "mlp")});
    specs.push_back({"block_output", make_block_name(block_idx, "out")});
    specs.push_back({"vit_output", "vit_output"});

    return specs;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml> [ref_dir] [block_idx]\n", argv[0]);
        fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
        fprintf(stderr, "Default block_idx: 15\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";
    const int block_idx = argc >= 4 ? std::atoi(argv[3]) : 15;
    if (block_idx < 0 || block_idx >= 32) {
        fprintf(stderr, "invalid block_idx=%d\n", block_idx);
        return 1;
    }

    auto preprocessed = load_ref_f32(ref_dir + "/preprocessed");
    if (preprocessed.data.empty() || preprocessed.shape.size() != 4) {
        fprintf(stderr, "failed to load %s/preprocessed\n", ref_dir.c_str());
        return 1;
    }
    const int img_size = preprocessed.shape[2];

    const std::vector<tensor_spec> specs = default_specs_for_block(block_idx);
    std::vector<std::string> names;
    names.reserve(specs.size());
    for (const auto & spec : specs) {
        names.push_back(spec.name);
    }

    const std::string cpu_dir = "/tmp/sam3_cpu_vit_upstream";
    const std::string metal_dir = "/tmp/sam3_metal_vit_upstream";
    ensure_dir(cpu_dir);
    ensure_dir(metal_dir);

    backend_run_result cpu_run;
    backend_run_result metal_run;

    fprintf(stderr, "\n=== CPU block %d selective ViT run ===\n", block_idx);
    if (!run_backend(model_path, preprocessed.data, img_size, false, 8, block_idx, names, cpu_dir, cpu_run)) {
        return 1;
    }
    fprintf(stderr, "CPU wall time: %.1f ms\n", cpu_run.elapsed_ms);

    fprintf(stderr, "\n=== Metal block %d selective ViT run ===\n", block_idx);
    if (!run_backend(model_path, preprocessed.data, img_size, true, 8, block_idx, names, metal_dir, metal_run)) {
        return 1;
    }
    fprintf(stderr, "Metal wall time: %.1f ms\n", metal_run.elapsed_ms);

    fprintf(stderr, "\n=== CPU vs Metal block %d ===\n", block_idx);
    fprintf(stderr, "%-28s %-24s %-18s %14s %14s %12s\n",
            "semantic", "tensor", "shape", "max_abs_diff", "mean_abs_diff", "n_bad");

    for (const auto & spec : specs) {
        auto cpu = load_ref_f32(cpu_dir + "/" + spec.name);
        auto metal = load_ref_f32(metal_dir + "/" + spec.name);
        if (cpu.data.empty() || metal.data.empty()) {
            fprintf(stderr, "%-28s %-24s %-18s %14s %14s %12s\n",
                    spec.label.c_str(), spec.name.c_str(), "-", "load-fail", "load-fail", "-");
            continue;
        }

        const compare_result diff = compare_tensors(cpu.data.data(), metal.data.data(), cpu.numel(), 1e-4f);
        fprintf(stderr, "%-28s %-24s %-18s %14.6f %14.6f %12d\n",
                spec.label.c_str(),
                spec.name.c_str(),
                format_shape(cpu.shape).c_str(),
                diff.max_diff,
                diff.mean_diff,
                diff.n_bad);
    }

    ref_tensor_f32 norm_w;
    ref_tensor_f32 norm_b;
    if (!load_block_norm_params(model_path, block_idx, norm_w, norm_b)) {
        fprintf(stderr, "\nfailed to load norm1 params for block %d\n", block_idx);
        return 1;
    }

    const ref_tensor_f32 block_input = load_ref_f32(cpu_dir + "/" + specs[0].name);
    const ref_tensor_f32 cpu_norm_dump = load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "norm1"));
    const ref_tensor_f32 metal_norm_dump = load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "norm1"));
    if (!block_input.data.empty() && !cpu_norm_dump.data.empty() && !metal_norm_dump.data.empty()) {
        ggml_backend_t cpu_backend = create_backend(false, 8);
        ggml_backend_t metal_backend = create_backend(true, 8);

        const isolated_run_result iso_cpu = run_isolated_norm(cpu_backend, block_input, norm_w, norm_b);
        const isolated_run_result iso_metal = run_isolated_norm(metal_backend, block_input, norm_w, norm_b);

        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
        if (metal_backend) {
            ggml_backend_free(metal_backend);
        }

        if (!iso_cpu.output.empty() && !iso_metal.output.empty()) {
            const compare_result metal_vs_cpu = compare_tensors(
                    iso_cpu.output.data(), iso_metal.output.data(), (int) iso_cpu.output.size(), 1e-4f);
            const compare_result cpu_vs_dump = compare_tensors(
                    cpu_norm_dump.data.data(), iso_cpu.output.data(), cpu_norm_dump.numel(), 1e-4f);
            const compare_result metal_vs_dump = compare_tensors(
                    metal_norm_dump.data.data(), iso_metal.output.data(), metal_norm_dump.numel(), 1e-4f);

            fprintf(stderr, "\n=== Isolated block %d norm1 ===\n", block_idx);
            fprintf(stderr, "input shape: %s\n", format_shape(block_input.shape).c_str());
            fprintf(stderr, "isolated metal_vs_cpu: max_abs_diff=%.6f mean_abs_diff=%.6f n_bad=%d\n",
                    metal_vs_cpu.max_diff, metal_vs_cpu.mean_diff, metal_vs_cpu.n_bad);
            fprintf(stderr, "isolated cpu_vs_graph_dump: max_abs_diff=%.6f mean_abs_diff=%.6f n_bad=%d\n",
                    cpu_vs_dump.max_diff, cpu_vs_dump.mean_diff, cpu_vs_dump.n_bad);
            fprintf(stderr, "isolated metal_vs_graph_dump: max_abs_diff=%.6f mean_abs_diff=%.6f n_bad=%d\n",
                    metal_vs_dump.max_diff, metal_vs_dump.mean_diff, metal_vs_dump.n_bad);
        }
    }

    return 0;
}
