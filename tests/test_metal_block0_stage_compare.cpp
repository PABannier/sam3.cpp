#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct dump_item {
    const char * label;
    const char * name;
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

static std::vector<int> trim_shape(const int64_t ne[4]) {
    std::vector<int> shape = {
        (int) ne[0],
        (int) ne[1],
        (int) ne[2],
        (int) ne[3],
    };
    while (!shape.empty() && shape.back() == 1) {
        shape.pop_back();
    }
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
}

static bool dump_named_tensors(const sam3_state & state,
                               const std::vector<dump_item> & items,
                               const std::string & output_dir) {
    for (const auto & item : items) {
        if (!sam3_dump_state_tensor(state, item.name, output_dir + "/" + item.name)) {
            fprintf(stderr, "failed to dump %s\n", item.name);
            return false;
        }
    }
    return true;
}

static bool run_selective_vit(sam3_state & state,
                              const sam3_model & model,
                              const std::vector<float> & chw_data,
                              int img_size,
                              const std::vector<dump_item> & items,
                              const std::string & output_dir) {
    std::vector<std::string> names;
    names.reserve(items.size());
    for (const auto & item : items) {
        names.push_back(item.name);
    }

    if (!sam3_encode_vit_from_preprocessed_selective(state, model, chw_data.data(), img_size, names)) {
        return false;
    }
    return dump_named_tensors(state, items, output_dir);
}

static ref_tensor_f32 run_stage_or_die(const sam3_model & model,
                                       sam3_vit_block_stage stage,
                                       const ref_tensor_f32 & input,
                                       int n_threads) {
    int64_t in_ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < input.shape.size() && i < 4; ++i) {
        in_ne[i] = input.shape[i];
    }

    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block_stage(model, 0, stage, input.data.data(), in_ne, out.data, out_ne, n_threads)) {
        fprintf(stderr, "stage %d failed\n", (int) stage);
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static ref_tensor_f32 run_linear_host_ref_or_die(const sam3_model & model,
                                                 sam3_vit_block_stage stage,
                                                 const ref_tensor_f32 & input,
                                                 bool use_double_accum) {
    int64_t in_ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < input.shape.size() && i < 4; ++i) {
        in_ne[i] = input.shape[i];
    }

    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block_linear_host_ref(model, 0, stage, input.data.data(), in_ne, use_double_accum, out.data, out_ne)) {
        fprintf(stderr, "host linear ref stage %d failed\n", (int) stage);
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static ref_tensor_f32 run_block0_input_or_die(const sam3_model & model,
                                              const std::vector<float> & chw_data,
                                              int img_size,
                                              int n_threads) {
    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block0_input(model, chw_data.data(), img_size, out.data, out_ne, n_threads)) {
        fprintf(stderr, "block0 input prefix failed\n");
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static ref_tensor_f32 add_ref(const ref_tensor_f32 & a, const ref_tensor_f32 & b) {
    if (a.shape != b.shape || a.data.size() != b.data.size()) {
        fprintf(stderr, "add_ref: shape mismatch\n");
        std::exit(1);
    }
    ref_tensor_f32 out;
    out.shape = a.shape;
    out.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

static void print_diff_row(const char * label,
                           const ref_tensor_f32 & a,
                           const ref_tensor_f32 & b,
                           float atol = 1e-4f) {
    const compare_result r = compare_tensors(a.data.data(), b.data.data(), a.numel(), atol);
    fprintf(stderr, "%-22s %-18s %12.6f %12.6f %10d\n",
            label,
            format_shape(a.shape).c_str(),
            r.max_diff,
            r.mean_diff,
            r.n_bad);
}

static void print_graph_row(const dump_item & item,
                            const std::string & cpu_dir,
                            const std::string & metal_dir) {
    const ref_tensor_f32 cpu = load_ref_f32(cpu_dir + "/" + item.name);
    const ref_tensor_f32 metal = load_ref_f32(metal_dir + "/" + item.name);
    if (cpu.data.empty() || metal.data.empty()) {
        fprintf(stderr, "%-22s %-18s %12s %12s %10s\n",
                item.label, "-", "load-fail", "load-fail", "-");
        return;
    }
    print_diff_row(item.label, cpu, metal);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml> [ref_dir]\n", argv[0]);
        fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";

    const std::vector<dump_item> graph_items = {
        { "block_input",      "dbg_after_ln_pre"    },
        { "qkv_proj_output",  "dbg_block_0_qkv_proj"},
        { "attn_proj_output", "dbg_block_0_attn_proj"},
        { "resid1_output",    "dbg_block_0_resid1"  },
        { "mlp_output",       "dbg_block_0_mlp"     },
        { "block_output",     "dbg_block_0_out"     },
    };

    const std::string cpu_dir = "/tmp/sam3_block0_stage_cpu";
    const std::string metal_dir = "/tmp/sam3_block0_stage_metal";
    ensure_dir(cpu_dir);
    ensure_dir(metal_dir);

    const ref_tensor_f32 preprocessed = load_ref_f32(ref_dir + "/preprocessed");
    if (preprocessed.data.empty() || preprocessed.shape.size() != 4) {
        fprintf(stderr, "failed to load %s/preprocessed\n", ref_dir.c_str());
        return 1;
    }
    const int img_size = preprocessed.shape[2];
    const int n_threads = 8;

    sam3_params cpu_params;
    cpu_params.model_path = model_path;
    cpu_params.use_gpu = false;
    cpu_params.n_threads = n_threads;

    sam3_params metal_params = cpu_params;
    metal_params.use_gpu = true;

    auto cpu_model = sam3_load_model(cpu_params);
    auto metal_model = sam3_load_model(metal_params);
    if (!cpu_model || !metal_model) {
        fprintf(stderr, "failed to load CPU or Metal model\n");
        return 1;
    }

    auto cpu_state = sam3_create_state(*cpu_model, cpu_params);
    auto metal_state = sam3_create_state(*metal_model, metal_params);
    if (!cpu_state || !metal_state) {
        fprintf(stderr, "failed to create CPU or Metal state\n");
        return 1;
    }

    if (!run_selective_vit(*cpu_state, *cpu_model, preprocessed.data, img_size, graph_items, cpu_dir)) {
        fprintf(stderr, "CPU selective block-0 run failed\n");
        return 1;
    }
    if (!run_selective_vit(*metal_state, *metal_model, preprocessed.data, img_size, graph_items, metal_dir)) {
        fprintf(stderr, "Metal selective block-0 run failed\n");
        return 1;
    }

    fprintf(stderr, "\n=== Block-0 graph outputs: CPU vs Metal ===\n");
    fprintf(stderr, "%-22s %-18s %12s %12s %10s\n",
            "checkpoint", "shape", "max_abs_diff", "mean_abs", "n_bad");
    for (const auto & item : graph_items) {
        print_graph_row(item, cpu_dir, metal_dir);
    }

    const ref_tensor_f32 cpu_block_input = run_block0_input_or_die(*cpu_model, preprocessed.data, img_size, n_threads);
    const ref_tensor_f32 metal_block_input = run_block0_input_or_die(*metal_model, preprocessed.data, img_size, n_threads);

    sam3_tensor_info qkv_info = {};
    sam3_tensor_info proj_info = {};
    sam3_tensor_info mlp1_info = {};
    sam3_tensor_info rope_info = {};
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks.0.attn.qkv.weight", qkv_info);
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks.0.attn.proj.weight", proj_info);
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks.0.mlp.lin1.weight", mlp1_info);
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks.0.attn.freqs_cis", rope_info);

    fprintf(stderr, "\n=== Exact block-0 case ===\n");
    fprintf(stderr, "block input shape: %s\n", format_shape(cpu_block_input.shape).c_str());
    fprintf(stderr, "window attention: full_spatial=72x72 window=24 heads=%d head_dim=%d\n", 16, 64);
    fprintf(stderr, "qkv weight:  shape=[%lld,%lld] type=%d\n",
            (long long) qkv_info.ne[0], (long long) qkv_info.ne[1], qkv_info.type);
    fprintf(stderr, "proj weight: shape=[%lld,%lld] type=%d\n",
            (long long) proj_info.ne[0], (long long) proj_info.ne[1], proj_info.type);
    fprintf(stderr, "mlp fc1:     shape=[%lld,%lld] type=%d\n",
            (long long) mlp1_info.ne[0], (long long) mlp1_info.ne[1], mlp1_info.type);
    fprintf(stderr, "freqs_cis:   shape=[%lld,%lld,%lld] type=%d\n",
            (long long) rope_info.ne[0], (long long) rope_info.ne[1], (long long) rope_info.ne[2], rope_info.type);
    print_diff_row("block0_input_exact", cpu_block_input, metal_block_input);

    const ref_tensor_f32 cpu_norm1_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_NORM1, cpu_block_input, n_threads);
    const ref_tensor_f32 metal_norm1_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_NORM1, metal_block_input, n_threads);
    const ref_tensor_f32 cpu_win_part_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_WINDOW_PART, cpu_norm1_self, n_threads);
    const ref_tensor_f32 metal_win_part_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_WINDOW_PART, metal_norm1_self, n_threads);
    const ref_tensor_f32 cpu_qkv_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, cpu_win_part_self, n_threads);
    const ref_tensor_f32 metal_qkv_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, metal_win_part_self, n_threads);
    const ref_tensor_f32 cpu_attn_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_ATTN_CORE, cpu_qkv_self, n_threads);
    const ref_tensor_f32 metal_attn_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_ATTN_CORE, metal_qkv_self, n_threads);
    const ref_tensor_f32 cpu_attn_proj_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, cpu_attn_self, n_threads);
    const ref_tensor_f32 metal_attn_proj_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, metal_attn_self, n_threads);
    const ref_tensor_f32 cpu_unpart_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, cpu_attn_proj_self, n_threads);
    const ref_tensor_f32 metal_unpart_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, metal_attn_proj_self, n_threads);
    const ref_tensor_f32 cpu_resid1_self = add_ref(cpu_block_input, cpu_unpart_self);
    const ref_tensor_f32 metal_resid1_self = add_ref(metal_block_input, metal_unpart_self);
    const ref_tensor_f32 cpu_norm2_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_NORM2, cpu_resid1_self, n_threads);
    const ref_tensor_f32 metal_norm2_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_NORM2, metal_resid1_self, n_threads);
    const ref_tensor_f32 cpu_mlp_self = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP, cpu_norm2_self, n_threads);
    const ref_tensor_f32 metal_mlp_self = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_MLP, metal_norm2_self, n_threads);
    const ref_tensor_f32 cpu_block_self = add_ref(cpu_resid1_self, cpu_mlp_self);
    const ref_tensor_f32 metal_block_self = add_ref(metal_resid1_self, metal_mlp_self);

    const ref_tensor_f32 cpu_norm1_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_NORM1, cpu_block_input, n_threads);
    const ref_tensor_f32 metal_norm1_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_NORM1, cpu_block_input, n_threads);
    const ref_tensor_f32 cpu_win_part_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_WINDOW_PART, cpu_norm1_shared, n_threads);
    const ref_tensor_f32 metal_win_part_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_WINDOW_PART, cpu_norm1_shared, n_threads);
    const ref_tensor_f32 cpu_qkv_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, cpu_win_part_shared, n_threads);
    const ref_tensor_f32 metal_qkv_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, cpu_win_part_shared, n_threads);
    const ref_tensor_f32 cpu_attn_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_ATTN_CORE, cpu_qkv_shared, n_threads);
    const ref_tensor_f32 metal_attn_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_ATTN_CORE, cpu_qkv_shared, n_threads);
    const ref_tensor_f32 cpu_attn_proj_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, cpu_attn_shared, n_threads);
    const ref_tensor_f32 metal_attn_proj_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, cpu_attn_shared, n_threads);
    const ref_tensor_f32 cpu_unpart_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, cpu_attn_proj_shared, n_threads);
    const ref_tensor_f32 metal_unpart_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, cpu_attn_proj_shared, n_threads);
    const ref_tensor_f32 cpu_resid1_shared = add_ref(cpu_block_input, cpu_unpart_shared);
    const ref_tensor_f32 metal_resid1_shared = add_ref(cpu_block_input, metal_unpart_shared);
    const ref_tensor_f32 cpu_norm2_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_NORM2, cpu_resid1_shared, n_threads);
    const ref_tensor_f32 metal_norm2_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_NORM2, cpu_resid1_shared, n_threads);
    const ref_tensor_f32 cpu_mlp_fc1_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP_FC1, cpu_norm2_shared, n_threads);
    const ref_tensor_f32 metal_mlp_fc1_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_MLP_FC1, cpu_norm2_shared, n_threads);
    const ref_tensor_f32 cpu_mlp_gelu_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP_GELU, cpu_mlp_fc1_shared, n_threads);
    const ref_tensor_f32 metal_mlp_gelu_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_MLP_GELU, cpu_mlp_fc1_shared, n_threads);
    const ref_tensor_f32 cpu_mlp_fc2_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP_FC2, cpu_mlp_gelu_shared, n_threads);
    const ref_tensor_f32 metal_mlp_fc2_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_MLP_FC2, cpu_mlp_gelu_shared, n_threads);
    const ref_tensor_f32 cpu_mlp_shared = run_stage_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP, cpu_norm2_shared, n_threads);
    const ref_tensor_f32 metal_mlp_shared = run_stage_or_die(*metal_model, SAM3_VIT_BLOCK_STAGE_MLP, cpu_norm2_shared, n_threads);
    const ref_tensor_f32 cpu_block_shared = add_ref(cpu_resid1_shared, cpu_mlp_shared);
    const ref_tensor_f32 metal_block_shared = add_ref(cpu_resid1_shared, metal_mlp_shared);

    const std::map<std::string, ref_tensor_f32> cpu_graph = {
        {"qkv_proj_output", load_ref_f32(cpu_dir + "/dbg_block_0_qkv_proj")},
        {"attn_proj_output", load_ref_f32(cpu_dir + "/dbg_block_0_attn_proj")},
        {"resid1_output", load_ref_f32(cpu_dir + "/dbg_block_0_resid1")},
        {"mlp_output", load_ref_f32(cpu_dir + "/dbg_block_0_mlp")},
        {"block_output", load_ref_f32(cpu_dir + "/dbg_block_0_out")},
    };
    const std::map<std::string, ref_tensor_f32> metal_graph = {
        {"qkv_proj_output", load_ref_f32(metal_dir + "/dbg_block_0_qkv_proj")},
        {"attn_proj_output", load_ref_f32(metal_dir + "/dbg_block_0_attn_proj")},
        {"resid1_output", load_ref_f32(metal_dir + "/dbg_block_0_resid1")},
        {"mlp_output", load_ref_f32(metal_dir + "/dbg_block_0_mlp")},
        {"block_output", load_ref_f32(metal_dir + "/dbg_block_0_out")},
    };

    fprintf(stderr, "\n=== Exact isolated stages on shared canonical CPU inputs ===\n");
    fprintf(stderr, "%-22s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("norm1_output", cpu_norm1_shared, metal_norm1_shared);
    print_diff_row("win_part_output", cpu_win_part_shared, metal_win_part_shared);
    print_diff_row("qkv_proj_output", cpu_qkv_shared, metal_qkv_shared);
    print_diff_row("attn_core_output", cpu_attn_shared, metal_attn_shared);
    print_diff_row("attn_proj_output", cpu_attn_proj_shared, metal_attn_proj_shared);
    print_diff_row("win_unpart_output", cpu_unpart_shared, metal_unpart_shared);
    print_diff_row("resid1_output", cpu_resid1_shared, metal_resid1_shared);
    print_diff_row("norm2_output", cpu_norm2_shared, metal_norm2_shared);
    print_diff_row("mlp_output", cpu_mlp_shared, metal_mlp_shared);
    print_diff_row("block_output", cpu_block_shared, metal_block_shared);

    const ref_tensor_f32 host_qkv_f64 = run_linear_host_ref_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, cpu_win_part_shared, true);
    const ref_tensor_f32 host_attn_proj_f64 = run_linear_host_ref_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, cpu_attn_shared, true);
    const ref_tensor_f32 host_mlp_fc1_f64 = run_linear_host_ref_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP_FC1, cpu_norm2_shared, true);
    const ref_tensor_f32 host_mlp_fc2_f64 = run_linear_host_ref_or_die(*cpu_model, SAM3_VIT_BLOCK_STAGE_MLP_FC2, cpu_mlp_gelu_shared, true);

    fprintf(stderr, "\n=== Block-0 host linear refs (double accumulation) ===\n");
    fprintf(stderr, "%-22s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("qkv_cpu_vs_host", cpu_qkv_shared, host_qkv_f64);
    print_diff_row("qkv_mtl_vs_host", metal_qkv_shared, host_qkv_f64);
    print_diff_row("attn_proj_cpu_vs_host", cpu_attn_proj_shared, host_attn_proj_f64);
    print_diff_row("attn_proj_mtl_vs_host", metal_attn_proj_shared, host_attn_proj_f64);
    print_diff_row("mlp_fc1_cpu_vs_host", cpu_mlp_fc1_shared, host_mlp_fc1_f64);
    print_diff_row("mlp_fc1_mtl_vs_host", metal_mlp_fc1_shared, host_mlp_fc1_f64);
    print_diff_row("mlp_fc2_cpu_vs_host", cpu_mlp_fc2_shared, host_mlp_fc2_f64);
    print_diff_row("mlp_fc2_mtl_vs_host", metal_mlp_fc2_shared, host_mlp_fc2_f64);

    fprintf(stderr, "\n=== Block-0 MLP split on shared canonical CPU inputs ===\n");
    fprintf(stderr, "%-22s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("mlp_fc1_output", cpu_mlp_fc1_shared, metal_mlp_fc1_shared);
    print_diff_row("mlp_gelu_output", cpu_mlp_gelu_shared, metal_mlp_gelu_shared);
    print_diff_row("mlp_fc2_output", cpu_mlp_fc2_shared, metal_mlp_fc2_shared);

    fprintf(stderr, "\n=== Isolated stage self-consistency vs graph outputs ===\n");
    fprintf(stderr, "%-22s %-18s %-24s %-24s\n",
            "stage", "shape", "cpu_iso_vs_cpu_graph", "metal_iso_vs_metal_graph");

    auto print_self_row = [&](const char * label,
                              const ref_tensor_f32 & cpu_iso,
                              const ref_tensor_f32 & metal_iso,
                              const char * graph_key) {
        const auto cpu_it = cpu_graph.find(graph_key);
        const auto metal_it = metal_graph.find(graph_key);
        if (cpu_it == cpu_graph.end() || metal_it == metal_graph.end()) {
            fprintf(stderr, "%-22s %-18s %-24s %-24s\n",
                    label, "-", "missing", "missing");
            return;
        }

        const compare_result cpu_r = compare_tensors(cpu_iso.data.data(), cpu_it->second.data.data(), cpu_iso.numel(), 1e-4f);
        const compare_result metal_r = compare_tensors(metal_iso.data.data(), metal_it->second.data.data(), metal_iso.numel(), 1e-4f);

        char cpu_buf[64];
        char metal_buf[64];
        snprintf(cpu_buf, sizeof(cpu_buf), "max=%.6f mean=%.6f", cpu_r.max_diff, cpu_r.mean_diff);
        snprintf(metal_buf, sizeof(metal_buf), "max=%.6f mean=%.6f", metal_r.max_diff, metal_r.mean_diff);
        fprintf(stderr, "%-22s %-18s %-24s %-24s\n",
                label,
                format_shape(cpu_iso.shape).c_str(),
                cpu_buf,
                metal_buf);
    };

    print_self_row("qkv_proj_output", cpu_qkv_self, metal_qkv_self, "qkv_proj_output");
    print_self_row("attn_proj_output", cpu_attn_proj_self, metal_attn_proj_self, "attn_proj_output");
    print_self_row("resid1_output", cpu_resid1_self, metal_resid1_self, "resid1_output");
    print_self_row("mlp_output", cpu_mlp_self, metal_mlp_self, "mlp_output");
    print_self_row("block_output", cpu_block_self, metal_block_self, "block_output");

    return 0;
}
