#define _USE_MATH_DEFINES

// Phase 3 Numerical Audit — test ViT backbone, neck, sinusoidal PE, RoPE
// against Python reference tensors dumped by dump_phase3_reference.py.
//
// Usage:
//   ./test_phase3 tests/ref_phase3 models/sam3-f16.ggml tests/test_random.jpg
//
// Tests:
//   1. RoPE computation (pure math, no model needed)
//   2. Sinusoidal PE computation (pure math, no model needed)
//   3. Full image encoding: preprocessing → ViT → neck → PE
//      Compares intermediate and final outputs against Python references.

#include "sam3.h"
#include "test_fs.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════════

struct ref_tensor {
    std::vector<float> data;
    std::vector<int> shape;
    int numel() const {
        int n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

static ref_tensor load_ref(const std::string & path) {
    ref_tensor t;
    {
        std::ifstream f(path + ".shape");
        if (!f) {
            fprintf(stderr, "  [SKIP] %s.shape not found\n", path.c_str());
            return t;
        }
        std::string line;
        std::getline(f, line);
        size_t pos = 0;
        while (pos < line.size()) {
            size_t end = line.find(',', pos);
            if (end == std::string::npos) end = line.size();
            t.shape.push_back(std::stoi(line.substr(pos, end - pos)));
            pos = end + 1;
        }
    }
    {
        std::ifstream f(path + ".bin", std::ios::binary);
        if (!f) {
            fprintf(stderr, "  [SKIP] %s.bin not found\n", path.c_str());
            t.shape.clear();
            return t;
        }
        t.data.resize(t.numel());
        f.read(reinterpret_cast<char *>(t.data.data()), t.numel() * sizeof(float));
    }
    return t;
}

struct compare_result {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    double cosine_sim = 0.0;
    int n_elements = 0;
    int n_bad = 0;  // elements exceeding tolerance
};

static compare_result compare_tensors(const float * a, const float * b, int n,
                                       float atol = 1e-4f) {
    compare_result r;
    r.n_elements = n;
    double sum_diff = 0.0;
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > r.max_diff) r.max_diff = diff;
        if (diff > atol) r.n_bad++;
        sum_diff += diff;
        dot_ab += (double)a[i] * b[i];
        dot_aa += (double)a[i] * a[i];
        dot_bb += (double)b[i] * b[i];
    }
    r.mean_diff = (float)(sum_diff / n);
    if (dot_aa > 0 && dot_bb > 0) {
        r.cosine_sim = dot_ab / (sqrt(dot_aa) * sqrt(dot_bb));
    }
    return r;
}

static bool check(const std::string & name, const float * got, const ref_tensor & ref,
                   float atol, int & n_pass, int & n_fail) {
    if (ref.data.empty()) {
        fprintf(stderr, "  [SKIP] %s — no reference data\n", name.c_str());
        return true;
    }
    auto r = compare_tensors(got, ref.data.data(), ref.numel(), atol);
    bool ok = r.max_diff <= atol;
    fprintf(stderr, "  %s %-45s: max=%.6e mean=%.6e cos=%.8f bad=%d/%d (atol=%.1e)\n",
            ok ? "[PASS]" : "[FAIL]", name.c_str(), r.max_diff, r.mean_diff,
            r.cosine_sim, r.n_bad, r.n_elements, atol);
    if (ok) n_pass++; else n_fail++;
    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test 1: RoPE computation
// ═══════════════════════════════════════════════════════════════════════════════

static void test_rope(const std::string & ref_dir, int & n_pass, int & n_fail) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Test: RoPE Frequencies                  ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n");

    struct { int end; float scale; const char * name; } cases[] = {
        {24, 1.0f, "rope_window_real"},
        {72, 24.0f/72.0f, "rope_global_real"},
    };

    const int dim = 64;
    const float theta = 10000.0f;
    const int half_dim = dim / 4;  // 16

    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freqs[i] = 1.0f / powf(theta, (float)(i * 4) / (float)dim);
    }

    for (auto & tc : cases) {
        auto ref = load_ref(ref_dir + "/" + tc.name);
        if (ref.data.empty()) continue;

        const int N = tc.end * tc.end;
        std::vector<float> our_rope(N * dim);

        for (int idx = 0; idx < N; ++idx) {
            float t_x = (float)(idx % tc.end) * tc.scale;
            float t_y = static_cast<float>(idx / tc.end) * tc.scale; // NOLINT(bugprone-integer-division)

            for (int i = 0; i < half_dim; ++i) {
                float angle_x = t_x * freqs[i];
                our_rope[idx * dim + i * 2 + 0] = cosf(angle_x);
                our_rope[idx * dim + i * 2 + 1] = sinf(angle_x);
            }
            for (int i = 0; i < half_dim; ++i) {
                float angle_y = t_y * freqs[i];
                our_rope[idx * dim + half_dim * 2 + i * 2 + 0] = cosf(angle_y);
                our_rope[idx * dim + half_dim * 2 + i * 2 + 1] = sinf(angle_y);
            }
        }

        check(tc.name, our_rope.data(), ref, 1e-5f, n_pass, n_fail);
    }

    // Also compare against checkpoint freqs_cis to verify conversion is lossless
    for (int bi : {0, 7}) {
        std::string name = "ckpt_freqs_cis_block" + std::to_string(bi);
        auto ref_ckpt = load_ref(ref_dir + "/" + name);
        auto ref_computed = load_ref(ref_dir + "/" + (bi == 0 ? "rope_window_real" : "rope_global_real"));
        if (!ref_ckpt.data.empty() && !ref_computed.data.empty()) {
            auto r = compare_tensors(ref_ckpt.data.data(), ref_computed.data.data(),
                                      std::min(ref_ckpt.numel(), ref_computed.numel()), 1e-6f);
            bool ok = r.max_diff <= 1e-6f;
            fprintf(stderr, "  %s ckpt vs computed block %d: max=%.6e\n",
                    ok ? "[PASS]" : "[FAIL]", bi, r.max_diff);
            if (ok) n_pass++; else n_fail++;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test 2: Sinusoidal PE
// ═══════════════════════════════════════════════════════════════════════════════

static void test_sinusoidal_pe(const std::string & ref_dir, int & n_pass, int & n_fail) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Test: Sinusoidal PE                     ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n");

    struct { int H; int W; const char * name; } cases[] = {
        {288, 288, "pe_288"},
        {144, 144, "pe_144"},
        { 72,  72, "pe_72"},
        { 36,  36, "pe_36"},
    };

    for (auto & tc : cases) {
        auto ref = load_ref(ref_dir + "/" + tc.name);
        if (ref.data.empty()) continue;

        const int H = tc.H, W = tc.W, d_model = 256;
        const int half = d_model / 2;  // 128
        const float scale = 2.0f * (float)M_PI;
        const float temperature = 10000.0f;

        // Match Python PositionEmbeddingSine.forward() exactly:
        // y_embed = arange(1, H+1) / (H + eps) * scale
        // dim_t = temperature ** (2 * (arange(half) // 2) / half)
        // pos_y = y_embed / dim_t
        // stack(sin(even), cos(odd)).flatten → interleaved sin/cos
        // Output: [1, 256, H, W] (NCHW)

        std::vector<float> our_pe(d_model * H * W);
        const float eps = 1e-6f;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pos_y = ((float)(y + 1) / ((float)H + eps)) * scale;
                float pos_x = ((float)(x + 1) / ((float)W + eps)) * scale;

                for (int i = 0; i < half; ++i) {
                    int paired = (i / 2) * 2;  // 0,0,2,2,4,4,...
                    float dim_t = powf(temperature, (float)paired / (float)half);

                    float val_x, val_y;
                    if (i % 2 == 0) {
                        val_x = sinf(pos_x / dim_t);
                        val_y = sinf(pos_y / dim_t);
                    } else {
                        val_x = cosf(pos_x / dim_t);
                        val_y = cosf(pos_y / dim_t);
                    }

                    // PyTorch output is [1, 256, H, W]
                    // Channel layout: first 128 = pos_y, next 128 = pos_x
                    our_pe[i * H * W + y * W + x] = val_y;
                    our_pe[(i + half) * H * W + y * W + x] = val_x;
                }
            }
        }

        check(tc.name, our_pe.data(), ref, 1e-5f, n_pass, n_fail);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test 3: Full image encoding (model-dependent)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_encode_image(const std::string & model_path,
                               const std::string & image_path,
                               const std::string & ref_dir,
                               int & n_pass, int & n_fail) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Test: Full Image Encoding               ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n");

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "  FATAL: Failed to load model\n");
        n_fail++;
        return;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "  FATAL: Failed to create state\n");
        n_fail++;
        return;
    }

    auto img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "  FATAL: Failed to load image\n");
        n_fail++;
        return;
    }

    bool ok = sam3_encode_image(*state, *model, img);
    if (!ok) {
        fprintf(stderr, "  FATAL: sam3_encode_image failed\n");
        n_fail++;
        return;
    }

    fprintf(stderr, "  Image encoding completed successfully\n");

    std::string dump_dir = ref_dir + "/cpp_out_phase3";
    ensure_dir(dump_dir);

    // ── Helper: dump and compare a state tensor ──────────────────────────
    // C++ tensors are in ggml layout. We need to transpose to PyTorch layout.

    // For NHWC tensors (ViT intermediates): ggml [E, W, H, B] → PyTorch [B, H, W, E]
    auto compare_nhwc = [&](const std::string & cpp_name, const std::string & ref_name,
                            float atol) {
        bool dumped = sam3_dump_state_tensor(*state, cpp_name, dump_dir + "/" + cpp_name);
        if (!dumped) {
            fprintf(stderr, "  [SKIP] %s — tensor '%s' not available in state\n",
                    ref_name.c_str(), cpp_name.c_str());
            return;
        }
        auto ref_t = load_ref(ref_dir + "/" + ref_name);
        auto cpp_t = load_ref(dump_dir + "/" + cpp_name);
        if (ref_t.data.empty() || cpp_t.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — data not available\n", ref_name.c_str());
            return;
        }
        // ggml [E, W, H, ...] → [H, W, E] (ignoring batch)
        int E = cpp_t.shape[0];
        int nW = cpp_t.shape[1];
        int nH = cpp_t.shape.size() > 2 ? cpp_t.shape[2] : 1;
        // ref is [1, H, W, E] or [B, H, W, E]
        std::vector<float> transposed(cpp_t.numel());
        for (int h = 0; h < nH; ++h)
            for (int w = 0; w < nW; ++w)
                for (int e = 0; e < E; ++e)
                    transposed[h * nW * E + w * E + e] = cpp_t.data[e + w * E + h * E * nW];

        check(ref_name + " (NHWC)", transposed.data(), ref_t, atol, n_pass, n_fail);
    };

    // For NCHW tensors (neck outputs): ggml [C, W, H, B] → PyTorch [1, C, H, W]
    auto compare_nchw = [&](const std::string & cpp_name, const std::string & ref_name,
                            float atol) {
        bool dumped = sam3_dump_state_tensor(*state, cpp_name, dump_dir + "/" + cpp_name);
        if (!dumped) {
            fprintf(stderr, "  [SKIP] %s — tensor '%s' not available in state\n",
                    ref_name.c_str(), cpp_name.c_str());
            return;
        }
        auto ref_t = load_ref(ref_dir + "/" + ref_name);
        auto cpp_t = load_ref(dump_dir + "/" + cpp_name);
        if (ref_t.data.empty() || cpp_t.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — data not available\n", ref_name.c_str());
            return;
        }
        // ggml [C, W, H, ...]: element (c,w,h) at c + w*C + h*C*W
        // PyTorch [1, C, H, W]: element (0,c,h,w) at c*H*W + h*W + w
        int C = cpp_t.shape[0];
        int nW = cpp_t.shape[1];
        int nH = cpp_t.shape.size() > 2 ? cpp_t.shape[2] : 1;
        std::vector<float> transposed(cpp_t.numel());
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < nH; ++h)
                for (int w = 0; w < nW; ++w)
                    transposed[c * nH * nW + h * nW + w] = cpp_t.data[c + w * C + h * C * nW];

        check(ref_name + " (NCHW)", transposed.data(), ref_t, atol, n_pass, n_fail);
    };

    // ── Compare ViT intermediates ────────────────────────────────────────
    fprintf(stderr, "\n  --- ViT Intermediates ---\n");

    // Patch embed: max_diff ~0.3 due to different bilinear resize between C++ and Python.
    // This is the root cause of all downstream differences.
    compare_nhwc("dbg_patch_embed", "patch_embed", 0.5f);

    // After pos embed: same as patch_embed (pos embed loaded exactly)
    compare_nhwc("dbg_after_pos_embed", "after_pos_embed", 0.5f);

    // After ln_pre: amplified by layer norm (max ~1.2)
    compare_nhwc("dbg_after_ln_pre", "after_ln_pre", 2.0f);

    // Block outputs (error accumulates through transformer layers)
    compare_nhwc("dbg_block_0_out", "block_0_out", 2.0f);
    compare_nhwc("dbg_block_7_out", "block_7_out", 5.0f);

    // ViT final output
    sam3_dump_state_tensor(*state, "vit_output", dump_dir + "/vit_output");
    auto ref_vit = load_ref(ref_dir + "/vit_output_bchw");
    auto cpp_vit = load_ref(dump_dir + "/vit_output");
    if (!ref_vit.data.empty() && !cpp_vit.data.empty()) {
        int E = cpp_vit.shape[0], W = cpp_vit.shape[1];
        int H = (cpp_vit.shape.size() > 2) ? cpp_vit.shape[2] : 1;
        std::vector<float> transposed(cpp_vit.numel());
        for (int e = 0; e < E; ++e)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    transposed[e * H * W + h * W + w] = cpp_vit.data[e + w * E + h * E * W];
        // ViT output after 32 blocks: max_diff can be ~80 due to preprocessing diff
        // (C++ bilinear resize vs Python PIL resize) amplified through 32 transformer layers.
        // Cosine sim >0.995 confirms the ViT is computing the correct function.
        // The isolated test (Test 4) with identical input achieves cos=1.0 on f32.
        auto vit_r = compare_tensors(transposed.data(), ref_vit.data.data(), ref_vit.numel(), 100.0f);
        bool vit_ok = vit_r.cosine_sim > 0.995 && vit_r.mean_diff < 0.1f;
        fprintf(stderr, "  %s %-45s: max=%.6e mean=%.6e cos=%.8f (tol: cos>0.995, mean<0.1)\n",
                vit_ok ? "[PASS]" : "[FAIL]", "vit_output_bchw",
                vit_r.max_diff, vit_r.mean_diff, vit_r.cosine_sim);
        if (vit_ok) n_pass++; else n_fail++;
    }

    // ── Compare neck outputs ─────────────────────────────────────────────
    fprintf(stderr, "\n  --- Neck Outputs (Detector) ---\n");
    for (int i = 0; i < 4; ++i) {
        std::string name = "neck_det_" + std::to_string(i);
        compare_nchw(name, name, 5.0f);
    }

    // ── Compare sinusoidal PE from state ─────────────────────────────────
    // The state stores PE tensors as neck_det_pe_i.
    fprintf(stderr, "\n  --- Sinusoidal PE (from state) ---\n");
    int pe_sizes[] = {288, 144, 72, 36};
    for (int i = 0; i < 4; ++i) {
        std::string cpp_pe_name = "neck_det_pe_" + std::to_string(i);
        std::string ref_pe_name = "pe_" + std::to_string(pe_sizes[i]);
        sam3_dump_state_tensor(*state, cpp_pe_name, dump_dir + "/" + cpp_pe_name);
        auto ref_pe = load_ref(ref_dir + "/" + ref_pe_name);
        auto cpp_pe = load_ref(dump_dir + "/" + cpp_pe_name);
        if (!ref_pe.data.empty() && !cpp_pe.data.empty()) {
            // PE is stored as [D, W, H, 1] in ggml, ref is [1, 256, H, W] in PyTorch
            int D = cpp_pe.shape[0], pW = cpp_pe.shape[1];
            int pH = (cpp_pe.shape.size() > 2) ? cpp_pe.shape[2] : 1;
            std::vector<float> transposed(cpp_pe.numel());
            for (int d = 0; d < D; ++d)
                for (int h = 0; h < pH; ++h)
                    for (int w = 0; w < pW; ++w)
                        transposed[d * pH * pW + h * pW + w] = cpp_pe.data[d + w * D + h * D * pW];
            check(ref_pe_name + " (state)", transposed.data(), ref_pe, 1e-4f, n_pass, n_fail);
        }
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test 4: Isolated ViT numerics using Python-preprocessed input
//  This feeds the exact same preprocessed image to the C++ pipeline to isolate
//  ViT numerical accuracy from preprocessing differences.
// ═══════════════════════════════════════════════════════════════════════════════

static void test_encode_from_preprocessed(const std::string & model_path,
                                           const std::string & ref_dir,
                                           int & n_pass, int & n_fail) {
    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Test: ViT Numerics (Python preprocess)  ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n");

    // Load Python-preprocessed image (CHW layout, [1, 3, 1008, 1008])
    auto preproc = load_ref(ref_dir + "/preprocessed");
    if (preproc.data.empty()) {
        fprintf(stderr, "  [SKIP] No preprocessed reference data\n");
        return;
    }
    fprintf(stderr, "  Loaded preprocessed image: shape=[");
    for (size_t i = 0; i < preproc.shape.size(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", preproc.shape[i]);
    }
    fprintf(stderr, "] numel=%d\n", preproc.numel());

    // The preprocessed data is [1, 3, 1008, 1008] in PyTorch NCHW layout.
    // For ggml, the input tensor is [W=1008, H=1008, C=3, B=1].
    // ggml flat index: x + y*W + c*W*H = same as CHW[c*H*W + y*W + x].
    // So the CHW data can be uploaded directly.
    const float * chw_data = preproc.data.data();
    int img_size = preproc.shape[2];  // 1008

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "  FATAL: Failed to load model\n");
        n_fail++;
        return;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "  FATAL: Failed to create state\n");
        n_fail++;
        return;
    }

    bool ok = sam3_encode_image_from_preprocessed(*state, *model, chw_data, img_size);
    if (!ok) {
        fprintf(stderr, "  FATAL: sam3_encode_image_from_preprocessed failed\n");
        n_fail++;
        return;
    }

    std::string dump_dir = ref_dir + "/cpp_out_from_preproc";
    ensure_dir(dump_dir);

    // ── Compare ViT output ────────────────────────────────────────────────
    fprintf(stderr, "\n  --- ViT Output (same input) ---\n");
    sam3_dump_state_tensor(*state, "vit_output", dump_dir + "/vit_output");
    auto ref_vit = load_ref(ref_dir + "/vit_output_bchw");
    auto cpp_vit = load_ref(dump_dir + "/vit_output");
    if (!ref_vit.data.empty() && !cpp_vit.data.empty()) {
        int E = cpp_vit.shape[0], W = cpp_vit.shape[1];
        int H = (cpp_vit.shape.size() > 2) ? cpp_vit.shape[2] : 1;
        std::vector<float> transposed(cpp_vit.numel());
        for (int e = 0; e < E; ++e)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    transposed[e * H * W + h * W + w] = cpp_vit.data[e + w * E + h * E * W];

        auto vit_r = compare_tensors(transposed.data(), ref_vit.data.data(), ref_vit.numel(), 1e-2f);
        // With identical input and f32 weights, expect very high cosine similarity.
        // With f16 weights, expect cos > 0.999 and mean_diff < 0.05.
        bool vit_ok = vit_r.cosine_sim > 0.999 && vit_r.mean_diff < 0.05f;
        fprintf(stderr, "  %s %-45s: max=%.6e mean=%.6e cos=%.8f bad=%d/%d\n",
                vit_ok ? "[PASS]" : "[FAIL]", "vit_output (same input)",
                vit_r.max_diff, vit_r.mean_diff, vit_r.cosine_sim,
                vit_r.n_bad, vit_r.n_elements);
        if (vit_ok) n_pass++; else n_fail++;
    }

    // ── Compare neck outputs ─────────────────────────────────────────────
    fprintf(stderr, "\n  --- Neck Outputs (same input) ---\n");
    const char * neck_names[] = {"neck_det_0", "neck_det_1", "neck_det_2", "neck_det_3"};
    for (int i = 0; i < 4; ++i) {
        bool dumped = sam3_dump_state_tensor(*state, neck_names[i], dump_dir + "/" + neck_names[i]);
        if (!dumped) {
            fprintf(stderr, "  [SKIP] %s — not available\n", neck_names[i]);
            continue;
        }
        auto ref_t = load_ref(ref_dir + "/" + neck_names[i]);
        auto cpp_t = load_ref(dump_dir + "/" + neck_names[i]);
        if (ref_t.data.empty() || cpp_t.data.empty()) continue;

        int C = cpp_t.shape[0];
        int nW = cpp_t.shape[1];
        int nH = cpp_t.shape.size() > 2 ? cpp_t.shape[2] : 1;
        std::vector<float> transposed(cpp_t.numel());
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < nH; ++h)
                for (int w = 0; w < nW; ++w)
                    transposed[c * nH * nW + h * nW + w] = cpp_t.data[c + w * C + h * C * nW];

        auto r = compare_tensors(transposed.data(), ref_t.data.data(), ref_t.numel(), 1.0f);
        bool pass = r.cosine_sim > 0.999 && r.mean_diff < 0.05f;
        fprintf(stderr, "  %s %-45s: max=%.6e mean=%.6e cos=%.8f bad=%d/%d\n",
                pass ? "[PASS]" : "[FAIL]", neck_names[i],
                r.max_diff, r.mean_diff, r.cosine_sim, r.n_bad, r.n_elements);
        if (pass) n_pass++; else n_fail++;
    }

    // ── Compare tracker neck outputs ────────────────────────────────────
    fprintf(stderr, "\n  --- Tracker Neck Outputs (same input) ---\n");
    const char * trk_names[] = {"neck_trk_0", "neck_trk_1", "neck_trk_2", "neck_trk_3"};
    for (int i = 0; i < 4; ++i) {
        bool dumped = sam3_dump_state_tensor(*state, trk_names[i], dump_dir + "/" + trk_names[i]);
        if (!dumped) {
            fprintf(stderr, "  [SKIP] %s — not available\n", trk_names[i]);
            continue;
        }
        auto ref_t = load_ref(ref_dir + "/" + trk_names[i]);
        auto cpp_t = load_ref(dump_dir + "/" + trk_names[i]);
        if (ref_t.data.empty() || cpp_t.data.empty()) continue;

        int C = cpp_t.shape[0];
        int nW = cpp_t.shape[1];
        int nH = cpp_t.shape.size() > 2 ? cpp_t.shape[2] : 1;
        std::vector<float> transposed(cpp_t.numel());
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < nH; ++h)
                for (int w = 0; w < nW; ++w)
                    transposed[c * nH * nW + h * nW + w] = cpp_t.data[c + w * C + h * C * nW];

        auto r = compare_tensors(transposed.data(), ref_t.data.data(), ref_t.numel(), 1.0f);
        bool pass = r.cosine_sim > 0.999 && r.mean_diff < 0.05f;
        fprintf(stderr, "  %s %-45s: max=%.6e mean=%.6e cos=%.8f bad=%d/%d\n",
                pass ? "[PASS]" : "[FAIL]", trk_names[i],
                r.max_diff, r.mean_diff, r.cosine_sim, r.n_bad, r.n_elements);
        if (pass) n_pass++; else n_fail++;
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <ref_dir> [model.ggml] [image.jpg]\n", argv[0]);
        fprintf(stderr, "\nGenerate reference data first:\n");
        fprintf(stderr, "  uv run python tests/dump_phase3_reference.py \\\n");
        fprintf(stderr, "    --checkpoint raw_weights/sam3.pt --image data/test_image.jpg\n");
        return 1;
    }

    std::string ref_dir = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "";
    std::string image_path = (argc > 3) ? argv[3] : "";

    int n_pass = 0, n_fail = 0;

    // Test 1: RoPE (no model needed)
    test_rope(ref_dir, n_pass, n_fail);

    // Test 2: Sinusoidal PE (no model needed)
    test_sinusoidal_pe(ref_dir, n_pass, n_fail);

    // Test 3: Full encoding (needs model + image)
    if (!model_path.empty() && !image_path.empty()) {
        test_encode_image(model_path, image_path, ref_dir, n_pass, n_fail);
    } else {
        fprintf(stderr, "\n[SKIP] Full encoding test — no model/image provided\n");
    }

    // Test 4: Isolated ViT numerics from Python-preprocessed input (needs model)
    if (!model_path.empty()) {
        test_encode_from_preprocessed(model_path, ref_dir, n_pass, n_fail);
    } else {
        fprintf(stderr, "\n[SKIP] Isolated ViT test — no model provided\n");
    }

    fprintf(stderr, "\n═══════════════════════════════════════════\n");
    fprintf(stderr, "Phase 3 Audit Results: %d passed, %d failed\n", n_pass, n_fail);
    fprintf(stderr, "═══════════════════════════════════════════\n");
    return n_fail > 0 ? 1 : 0;
}
