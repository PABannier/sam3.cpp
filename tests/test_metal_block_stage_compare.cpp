#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct dump_item {
    const char * label;
    std::string  name;
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

static std::string make_block_name(int block_idx, const char * suffix) {
    char buf[64];
    snprintf(buf, sizeof(buf), "dbg_block_%d_%s", block_idx, suffix);
    return buf;
}

static bool dump_named_tensors(const sam3_state & state,
                               const std::vector<dump_item> & items,
                               const std::string & output_dir) {
    for (const auto & item : items) {
        if (!sam3_dump_state_tensor(state, item.name, output_dir + "/" + item.name)) {
            fprintf(stderr, "failed to dump %s\n", item.name.c_str());
            return false;
        }
    }
    return true;
}

static bool run_selective_vit(sam3_state & state,
                              const sam3_model & model,
                              const std::vector<float> & chw_data,
                              int img_size,
                              int debug_block_idx,
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
                                       int block_idx,
                                       sam3_vit_block_stage stage,
                                       const ref_tensor_f32 & input,
                                       int n_threads) {
    int64_t in_ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < input.shape.size() && i < 4; ++i) {
        in_ne[i] = input.shape[i];
    }

    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block_stage(model, block_idx, stage, input.data.data(), in_ne, out.data, out_ne, n_threads)) {
        fprintf(stderr, "stage %d for block %d failed\n", (int) stage, block_idx);
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static ref_tensor_f32 run_linear_host_ref_or_die(const sam3_model & model,
                                                 int block_idx,
                                                 sam3_vit_block_stage stage,
                                                 const ref_tensor_f32 & input,
                                                 bool use_double_accum) {
    int64_t in_ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < input.shape.size() && i < 4; ++i) {
        in_ne[i] = input.shape[i];
    }

    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block_linear_host_ref(
                model, block_idx, stage, input.data.data(), in_ne, use_double_accum, out.data, out_ne)) {
        fprintf(stderr, "host linear ref stage %d for block %d failed\n", (int) stage, block_idx);
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

static ref_tensor_f32 slice_first_columns(const ref_tensor_f32 & t, int n_cols) {
    if (t.shape.empty()) {
        fprintf(stderr, "slice_first_columns: empty tensor\n");
        std::exit(1);
    }

    const int ne0 = t.shape[0];
    const int total_cols = (int) t.data.size() / ne0;
    if (total_cols <= 0 || ne0 * total_cols != (int) t.data.size()) {
        fprintf(stderr, "slice_first_columns: invalid shape/data size\n");
        std::exit(1);
    }

    const int cols = n_cols < total_cols ? n_cols : total_cols;

    ref_tensor_f32 out;
    out.shape = {ne0, cols, 1, 1};
    out.data.resize((size_t) ne0 * cols);
    std::copy_n(t.data.begin(), (size_t) ne0 * cols, out.data.begin());
    return out;
}

static ref_tensor_f32 slice_column_range(const ref_tensor_f32 & t, int col0, int n_cols) {
    if (t.shape.empty()) {
        fprintf(stderr, "slice_column_range: empty tensor\n");
        std::exit(1);
    }

    const int ne0 = t.shape[0];
    const int total_cols = (int) t.data.size() / ne0;
    if (total_cols <= 0 || ne0 * total_cols != (int) t.data.size()) {
        fprintf(stderr, "slice_column_range: invalid shape/data size\n");
        std::exit(1);
    }

    if (col0 < 0) {
        col0 = 0;
    }
    if (col0 >= total_cols) {
        col0 = total_cols - 1;
    }
    const int cols = (col0 + n_cols <= total_cols) ? n_cols : (total_cols - col0);

    ref_tensor_f32 out;
    out.shape = {ne0, cols, 1, 1};
    out.data.resize((size_t) ne0 * cols);
    const float * src = t.data.data() + (size_t) ne0 * col0;
    std::copy_n(src, (size_t) ne0 * cols, out.data.begin());
    return out;
}

static compare_result print_diff_row_ex(const char * label,
                                        const ref_tensor_f32 & a,
                                        const ref_tensor_f32 & b,
                                        float atol = 1e-4f) {
    const compare_result r = compare_tensors(a.data.data(), b.data.data(), a.numel(), atol);
    fprintf(stderr, "%-26s %-18s %12.6f %12.6f %10d\n",
            label,
            format_shape(a.shape).c_str(),
            r.max_diff,
            r.mean_diff,
            r.n_bad);
    return r;
}

static void print_diff_row(const char * label,
                           const ref_tensor_f32 & a,
                           const ref_tensor_f32 & b,
                           float atol = 1e-4f) {
    (void) print_diff_row_ex(label, a, b, atol);
}

static void print_graph_row(const dump_item & item,
                            const std::string & cpu_dir,
                            const std::string & metal_dir) {
    const ref_tensor_f32 cpu = load_ref_f32(cpu_dir + "/" + item.name);
    const ref_tensor_f32 metal = load_ref_f32(metal_dir + "/" + item.name);
    if (cpu.data.empty() || metal.data.empty()) {
        fprintf(stderr, "%-26s %-18s %12s %12s %10s\n",
                item.label, "-", "load-fail", "load-fail", "-");
        return;
    }
    print_diff_row(item.label, cpu, metal);
}

static bool is_global_block(const sam3_model & model, int block_idx, sam3_tensor_info & rope_info) {
    const std::string rope_name = "vit.blocks." + std::to_string(block_idx) + ".attn.freqs_cis";
    if (!sam3_get_model_tensor_info(model, rope_name, rope_info)) {
        fprintf(stderr, "failed to load %s info\n", rope_name.c_str());
        std::exit(1);
    }
    return rope_info.ne[2] == 5184;
}

struct stage_bundle {
    ref_tensor_f32 norm1;
    ref_tensor_f32 attn_input;
    ref_tensor_f32 qkv;
    ref_tensor_f32 attn;
    ref_tensor_f32 attn_proj;
    ref_tensor_f32 post_attn;
    ref_tensor_f32 resid1;
    ref_tensor_f32 norm2;
    ref_tensor_f32 mlp_fc1;
    ref_tensor_f32 mlp_gelu;
    ref_tensor_f32 mlp_fc2;
    ref_tensor_f32 mlp;
    ref_tensor_f32 block;
};

static stage_bundle run_block_bundle(const sam3_model & model,
                                     int block_idx,
                                     bool is_global,
                                     const ref_tensor_f32 & block_input,
                                     int n_threads) {
    stage_bundle out;

    out.norm1 = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_NORM1, block_input, n_threads);
    out.attn_input = is_global
            ? out.norm1
            : run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_WINDOW_PART, out.norm1, n_threads);
    out.qkv = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, out.attn_input, n_threads);
    out.attn = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_ATTN_CORE, out.qkv, n_threads);
    out.attn_proj = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, out.attn, n_threads);
    out.post_attn = is_global
            ? out.attn_proj
            : run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, out.attn_proj, n_threads);
    out.resid1 = add_ref(block_input, out.post_attn);
    out.norm2 = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_NORM2, out.resid1, n_threads);
    out.mlp_fc1 = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC1, out.norm2, n_threads);
    out.mlp_gelu = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_GELU, out.mlp_fc1, n_threads);
    out.mlp_fc2 = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC2, out.mlp_gelu, n_threads);
    out.mlp = run_stage_or_die(model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP, out.norm2, n_threads);
    out.block = add_ref(out.resid1, out.mlp);

    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml> [ref_dir] [block_idx]\n", argv[0]);
        fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
        fprintf(stderr, "Default block_idx: 16\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";
    const int block_idx = argc >= 4 ? std::atoi(argv[3]) : 16;

    const std::string input_name = block_idx == 0 ? "dbg_after_ln_pre" : make_block_name(block_idx - 1, "out");
    const std::vector<dump_item> graph_items = {
        { "block_input",      input_name },
        { "qkv_proj_output",  make_block_name(block_idx, "qkv_proj") },
        { "attn_proj_output", make_block_name(block_idx, "attn_proj") },
        { "resid1_output",    make_block_name(block_idx, "resid1") },
        { "mlp_output",       make_block_name(block_idx, "mlp") },
        { "block_output",     make_block_name(block_idx, "out") },
    };

    const std::string cpu_dir = "/tmp/sam3_block_stage_cpu";
    const std::string metal_dir = "/tmp/sam3_block_stage_metal";
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

    if (!run_selective_vit(*cpu_state, *cpu_model, preprocessed.data, img_size, block_idx, graph_items, cpu_dir)) {
        fprintf(stderr, "CPU selective block run failed\n");
        return 1;
    }
    if (!run_selective_vit(*metal_state, *metal_model, preprocessed.data, img_size, block_idx, graph_items, metal_dir)) {
        fprintf(stderr, "Metal selective block run failed\n");
        return 1;
    }

    fprintf(stderr, "\n=== Block-%d graph outputs: CPU vs Metal ===\n", block_idx);
    fprintf(stderr, "%-26s %-18s %12s %12s %10s\n",
            "checkpoint", "shape", "max_abs_diff", "mean_abs", "n_bad");
    for (const auto & item : graph_items) {
        print_graph_row(item, cpu_dir, metal_dir);
    }

    const ref_tensor_f32 cpu_block_input = load_ref_f32(cpu_dir + "/" + input_name);
    const ref_tensor_f32 metal_block_input = load_ref_f32(metal_dir + "/" + input_name);
    if (cpu_block_input.data.empty() || metal_block_input.data.empty()) {
        fprintf(stderr, "failed to load canonical block input %s\n", input_name.c_str());
        return 1;
    }

    sam3_tensor_info rope_info = {};
    const bool global = is_global_block(*cpu_model, block_idx, rope_info);

    sam3_tensor_info qkv_info = {};
    sam3_tensor_info proj_info = {};
    sam3_tensor_info mlp1_info = {};
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks." + std::to_string(block_idx) + ".attn.qkv.weight", qkv_info);
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks." + std::to_string(block_idx) + ".attn.proj.weight", proj_info);
    sam3_get_model_tensor_info(*cpu_model, "vit.blocks." + std::to_string(block_idx) + ".mlp.lin1.weight", mlp1_info);

    fprintf(stderr, "\n=== Exact block-%d case ===\n", block_idx);
    fprintf(stderr, "block input shape: %s\n", format_shape(cpu_block_input.shape).c_str());
    fprintf(stderr, "attention type:    %s\n", global ? "global" : "windowed");
    fprintf(stderr, "qkv weight:        shape=[%lld,%lld] type=%d\n",
            (long long) qkv_info.ne[0], (long long) qkv_info.ne[1], qkv_info.type);
    fprintf(stderr, "proj weight:       shape=[%lld,%lld] type=%d\n",
            (long long) proj_info.ne[0], (long long) proj_info.ne[1], proj_info.type);
    fprintf(stderr, "mlp fc1 weight:    shape=[%lld,%lld] type=%d\n",
            (long long) mlp1_info.ne[0], (long long) mlp1_info.ne[1], mlp1_info.type);
    fprintf(stderr, "freqs_cis:         shape=[%lld,%lld,%lld] type=%d\n",
            (long long) rope_info.ne[0], (long long) rope_info.ne[1], (long long) rope_info.ne[2], rope_info.type);
    print_diff_row("block_input_exact", cpu_block_input, metal_block_input);

    const stage_bundle cpu_shared = run_block_bundle(*cpu_model, block_idx, global, cpu_block_input, n_threads);
    const stage_bundle metal_shared = run_block_bundle(*metal_model, block_idx, global, cpu_block_input, n_threads);
    const stage_bundle cpu_self = run_block_bundle(*cpu_model, block_idx, global, cpu_block_input, n_threads);
    const stage_bundle metal_self = run_block_bundle(*metal_model, block_idx, global, metal_block_input, n_threads);

    fprintf(stderr, "\n=== Exact isolated stages on shared canonical CPU inputs ===\n");
    fprintf(stderr, "%-26s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("norm1_output", cpu_shared.norm1, metal_shared.norm1);
    print_diff_row(global ? "attn_input_output" : "window_part_output", cpu_shared.attn_input, metal_shared.attn_input);
    const compare_result qkv_diff = print_diff_row_ex("qkv_proj_output", cpu_shared.qkv, metal_shared.qkv);
    const compare_result attn_diff = print_diff_row_ex("attn_core_output", cpu_shared.attn, metal_shared.attn);
    const compare_result attn_proj_diff = print_diff_row_ex("attn_proj_output", cpu_shared.attn_proj, metal_shared.attn_proj);
    print_diff_row(global ? "post_attn_output" : "window_unpart_output", cpu_shared.post_attn, metal_shared.post_attn);
    print_diff_row("resid1_output", cpu_shared.resid1, metal_shared.resid1);
    print_diff_row("norm2_output", cpu_shared.norm2, metal_shared.norm2);
    print_diff_row("mlp_fc1_output", cpu_shared.mlp_fc1, metal_shared.mlp_fc1);
    print_diff_row("mlp_gelu_output", cpu_shared.mlp_gelu, metal_shared.mlp_gelu);
    const compare_result mlp_fc2_diff = print_diff_row_ex("mlp_fc2_output", cpu_shared.mlp_fc2, metal_shared.mlp_fc2);
    print_diff_row("mlp_output", cpu_shared.mlp, metal_shared.mlp);
    print_diff_row("block_output", cpu_shared.block, metal_shared.block);

    auto print_worst_loc = [](const char * label, const ref_tensor_f32 & t, const compare_result & diff) {
        const int ne0 = t.shape.empty() ? 0 : t.shape[0];
        const int worst_col = ne0 > 0 ? diff.worst_index / ne0 : -1;
        const int worst_row = ne0 > 0 ? diff.worst_index % ne0 : -1;
        fprintf(stderr, "%s worst: col=%d row=%d cpu=%.6f metal=%.6f max_abs=%.6f\n",
                label, worst_col, worst_row, diff.worst_a, diff.worst_b, diff.max_diff);
    };

    fprintf(stderr, "\n=== Worst-element locations on shared canonical CPU inputs ===\n");
    print_worst_loc("qkv_proj_output", cpu_shared.qkv, qkv_diff);
    print_worst_loc("attn_core_output", cpu_shared.attn, attn_diff);
    print_worst_loc("attn_proj_output", cpu_shared.attn_proj, attn_proj_diff);
    print_worst_loc("mlp_fc2_output", cpu_shared.mlp_fc2, mlp_fc2_diff);

    const int linear_slice_cols = 32;
    const ref_tensor_f32 qkv_in_slice = slice_first_columns(cpu_shared.attn_input, linear_slice_cols);
    const ref_tensor_f32 qkv_cpu_slice = slice_first_columns(cpu_shared.qkv, linear_slice_cols);
    const ref_tensor_f32 qkv_mtl_slice = slice_first_columns(metal_shared.qkv, linear_slice_cols);
    const ref_tensor_f32 attn_proj_in_slice = slice_first_columns(cpu_shared.attn, linear_slice_cols);
    const ref_tensor_f32 attn_proj_cpu_slice = slice_first_columns(cpu_shared.attn_proj, linear_slice_cols);
    const ref_tensor_f32 attn_proj_mtl_slice = slice_first_columns(metal_shared.attn_proj, linear_slice_cols);
    const ref_tensor_f32 mlp_fc1_in_slice = slice_first_columns(cpu_shared.norm2, linear_slice_cols);
    const ref_tensor_f32 mlp_fc1_cpu_slice = slice_first_columns(cpu_shared.mlp_fc1, linear_slice_cols);
    const ref_tensor_f32 mlp_fc1_mtl_slice = slice_first_columns(metal_shared.mlp_fc1, linear_slice_cols);
    const ref_tensor_f32 mlp_fc2_in_slice = slice_first_columns(cpu_shared.mlp_gelu, linear_slice_cols);
    const ref_tensor_f32 mlp_fc2_cpu_slice = slice_first_columns(cpu_shared.mlp_fc2, linear_slice_cols);
    const ref_tensor_f32 mlp_fc2_mtl_slice = slice_first_columns(metal_shared.mlp_fc2, linear_slice_cols);

    const ref_tensor_f32 host_qkv_f32 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, qkv_in_slice, false);
    const ref_tensor_f32 host_qkv_f64 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, qkv_in_slice, true);
    const ref_tensor_f32 host_attn_proj_f32 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, attn_proj_in_slice, false);
    const ref_tensor_f32 host_attn_proj_f64 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, attn_proj_in_slice, true);
    const ref_tensor_f32 host_mlp_fc1_f32 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC1, mlp_fc1_in_slice, false);
    const ref_tensor_f32 host_mlp_fc1_f64 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC1, mlp_fc1_in_slice, true);
    const ref_tensor_f32 host_mlp_fc2_f32 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC2, mlp_fc2_in_slice, false);
    const ref_tensor_f32 host_mlp_fc2_f64 = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC2, mlp_fc2_in_slice, true);

    fprintf(stderr, "\n=== Exact block-%d linear host refs (first %d tokens) ===\n", block_idx, linear_slice_cols);
    fprintf(stderr, "%-26s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("qkv_cpu_vs_host_f32", qkv_cpu_slice, host_qkv_f32);
    print_diff_row("qkv_mtl_vs_host_f32", qkv_mtl_slice, host_qkv_f32);
    print_diff_row("qkv_cpu_vs_host_f64", qkv_cpu_slice, host_qkv_f64);
    print_diff_row("qkv_mtl_vs_host_f64", qkv_mtl_slice, host_qkv_f64);
    print_diff_row("attn_proj_cpu_vs_f32", attn_proj_cpu_slice, host_attn_proj_f32);
    print_diff_row("attn_proj_mtl_vs_f32", attn_proj_mtl_slice, host_attn_proj_f32);
    print_diff_row("attn_proj_cpu_vs_f64", attn_proj_cpu_slice, host_attn_proj_f64);
    print_diff_row("attn_proj_mtl_vs_f64", attn_proj_mtl_slice, host_attn_proj_f64);
    print_diff_row("mlp_fc1_cpu_vs_f32", mlp_fc1_cpu_slice, host_mlp_fc1_f32);
    print_diff_row("mlp_fc1_mtl_vs_f32", mlp_fc1_mtl_slice, host_mlp_fc1_f32);
    print_diff_row("mlp_fc1_cpu_vs_f64", mlp_fc1_cpu_slice, host_mlp_fc1_f64);
    print_diff_row("mlp_fc1_mtl_vs_f64", mlp_fc1_mtl_slice, host_mlp_fc1_f64);
    print_diff_row("mlp_fc2_cpu_vs_f32", mlp_fc2_cpu_slice, host_mlp_fc2_f32);
    print_diff_row("mlp_fc2_mtl_vs_f32", mlp_fc2_mtl_slice, host_mlp_fc2_f32);
    print_diff_row("mlp_fc2_cpu_vs_f64", mlp_fc2_cpu_slice, host_mlp_fc2_f64);
    print_diff_row("mlp_fc2_mtl_vs_f64", mlp_fc2_mtl_slice, host_mlp_fc2_f64);

    const int qkv_worst_col = qkv_diff.worst_index / cpu_shared.qkv.shape[0];
    const int attn_proj_worst_col = attn_proj_diff.worst_index / cpu_shared.attn_proj.shape[0];
    const int mlp_fc2_worst_col = mlp_fc2_diff.worst_index / cpu_shared.mlp_fc2.shape[0];

    const ref_tensor_f32 qkv_in_worst = slice_column_range(cpu_shared.attn_input, qkv_worst_col, 1);
    const ref_tensor_f32 qkv_cpu_worst = slice_column_range(cpu_shared.qkv, qkv_worst_col, 1);
    const ref_tensor_f32 qkv_mtl_worst = slice_column_range(metal_shared.qkv, qkv_worst_col, 1);
    const ref_tensor_f32 qkv_host_worst = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_QKV_PROJ, qkv_in_worst, false);

    const ref_tensor_f32 attn_proj_in_worst = slice_column_range(cpu_shared.attn, attn_proj_worst_col, 1);
    const ref_tensor_f32 attn_proj_cpu_worst = slice_column_range(cpu_shared.attn_proj, attn_proj_worst_col, 1);
    const ref_tensor_f32 attn_proj_mtl_worst = slice_column_range(metal_shared.attn_proj, attn_proj_worst_col, 1);
    const ref_tensor_f32 attn_proj_host_worst = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, attn_proj_in_worst, false);

    const ref_tensor_f32 mlp_fc2_in_worst = slice_column_range(cpu_shared.mlp_gelu, mlp_fc2_worst_col, 1);
    const ref_tensor_f32 mlp_fc2_cpu_worst = slice_column_range(cpu_shared.mlp_fc2, mlp_fc2_worst_col, 1);
    const ref_tensor_f32 mlp_fc2_mtl_worst = slice_column_range(metal_shared.mlp_fc2, mlp_fc2_worst_col, 1);
    const ref_tensor_f32 mlp_fc2_host_worst = run_linear_host_ref_or_die(*cpu_model, block_idx, SAM3_VIT_BLOCK_STAGE_MLP_FC2, mlp_fc2_in_worst, false);

    fprintf(stderr, "\n=== Worst-column host refs (single real token) ===\n");
    fprintf(stderr, "%-26s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");
    print_diff_row("qkv_cpu_vs_host_worst", qkv_cpu_worst, qkv_host_worst);
    print_diff_row("qkv_mtl_vs_host_worst", qkv_mtl_worst, qkv_host_worst);
    print_diff_row("attn_proj_cpu_vs_worst", attn_proj_cpu_worst, attn_proj_host_worst);
    print_diff_row("attn_proj_mtl_vs_worst", attn_proj_mtl_worst, attn_proj_host_worst);
    print_diff_row("mlp_fc2_cpu_vs_worst", mlp_fc2_cpu_worst, mlp_fc2_host_worst);
    print_diff_row("mlp_fc2_mtl_vs_worst", mlp_fc2_mtl_worst, mlp_fc2_host_worst);

    const std::map<std::string, ref_tensor_f32> cpu_graph = {
        {"qkv_proj_output", load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "qkv_proj"))},
        {"attn_proj_output", load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "attn_proj"))},
        {"resid1_output", load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "resid1"))},
        {"mlp_output", load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "mlp"))},
        {"block_output", load_ref_f32(cpu_dir + "/" + make_block_name(block_idx, "out"))},
    };
    const std::map<std::string, ref_tensor_f32> metal_graph = {
        {"qkv_proj_output", load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "qkv_proj"))},
        {"attn_proj_output", load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "attn_proj"))},
        {"resid1_output", load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "resid1"))},
        {"mlp_output", load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "mlp"))},
        {"block_output", load_ref_f32(metal_dir + "/" + make_block_name(block_idx, "out"))},
    };

    fprintf(stderr, "\n=== Isolated self-consistency vs graph outputs ===\n");
    fprintf(stderr, "%-26s %-18s %-24s %-24s\n",
            "stage", "shape", "cpu_iso_vs_cpu_graph", "metal_iso_vs_metal_graph");

    auto print_self_row = [&](const char * label,
                              const ref_tensor_f32 & cpu_iso,
                              const ref_tensor_f32 & metal_iso,
                              const char * graph_key) {
        const auto cpu_it = cpu_graph.find(graph_key);
        const auto metal_it = metal_graph.find(graph_key);
        if (cpu_it == cpu_graph.end() || metal_it == metal_graph.end()) {
            fprintf(stderr, "%-26s %-18s %-24s %-24s\n", label, "-", "missing", "missing");
            return;
        }

        const compare_result cpu_r = compare_tensors(cpu_iso.data.data(), cpu_it->second.data.data(), cpu_iso.numel(), 1e-4f);
        const compare_result metal_r = compare_tensors(metal_iso.data.data(), metal_it->second.data.data(), metal_iso.numel(), 1e-4f);

        char cpu_buf[64];
        char metal_buf[64];
        snprintf(cpu_buf, sizeof(cpu_buf), "max=%.6f mean=%.6f", cpu_r.max_diff, cpu_r.mean_diff);
        snprintf(metal_buf, sizeof(metal_buf), "max=%.6f mean=%.6f", metal_r.max_diff, metal_r.mean_diff);
        fprintf(stderr, "%-26s %-18s %-24s %-24s\n",
                label,
                format_shape(cpu_iso.shape).c_str(),
                cpu_buf,
                metal_buf);
    };

    print_self_row("qkv_proj_output", cpu_self.qkv, metal_self.qkv, "qkv_proj_output");
    print_self_row("attn_proj_output", cpu_self.attn_proj, metal_self.attn_proj, "attn_proj_output");
    print_self_row("resid1_output", cpu_self.resid1, metal_self.resid1, "resid1_output");
    print_self_row("mlp_output", cpu_self.mlp, metal_self.mlp, "mlp_output");
    print_self_row("block_output", cpu_self.block, metal_self.block, "block_output");

    return 0;
}
