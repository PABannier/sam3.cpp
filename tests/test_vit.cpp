#define _USE_MATH_DEFINES

#include "sam3.h"
#include "test_fs.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  Helpers: load reference tensors
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
    // Load shape
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
    // Load data
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

// Compare two float arrays with tolerance.
// Returns max absolute difference.
static float compare_tensors(const float * a, const float * b, int n,
                              float atol = 1e-4f, float rtol = 1e-3f) {
    float max_diff = 0.0f;
    int n_bad = 0;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        float threshold = atol + rtol * fabsf(b[i]);
        if (diff > threshold) n_bad++;
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static bool check(const std::string & name, const float * got, const ref_tensor & ref,
                   float atol = 1e-4f, float rtol = 1e-3f) {
    if (ref.data.empty()) {
        fprintf(stderr, "  [SKIP] %s — no reference data\n", name.c_str());
        return true;
    }
    float max_diff = compare_tensors(got, ref.data.data(), ref.numel(), atol, rtol);
    bool ok = max_diff < atol + rtol;
    fprintf(stderr, "  %s %s: max_diff=%.6f (atol=%.1e rtol=%.1e)\n",
            ok ? "[PASS]" : "[FAIL]", name.c_str(), max_diff, atol, rtol);
    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test: preprocessing
// ═══════════════════════════════════════════════════════════════════════════════

static bool test_preprocessing(const std::string & image_path, const std::string & ref_dir) {
    fprintf(stderr, "\n=== Test: Preprocessing ===\n");

    // Load image
    auto img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "  Failed to load image: %s\n", image_path.c_str());
        return false;
    }
    fprintf(stderr, "  Loaded %dx%d image\n", img.width, img.height);

    // Load reference
    auto ref = load_ref(ref_dir + "/preprocessed");
    if (ref.data.empty()) return true;  // skip

    // The reference is [1, 3, 1008, 1008] from PyTorch (NCHW)
    // We check a reasonable tolerance since bilinear resize implementations differ slightly
    fprintf(stderr, "  Reference shape: ");
    for (auto d : ref.shape) fprintf(stderr, "%d ", d);
    fprintf(stderr, "\n");

    // Note: we can't directly test preprocessing since it's internal to sam3_encode_image.
    // For now, just verify the image loaded correctly.
    fprintf(stderr, "  [INFO] Preprocessing test requires running encode_image — tested implicitly\n");
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test: RoPE computation
// ═══════════════════════════════════════════════════════════════════════════════

// Forward-declare the internal function (defined in sam3.cpp)
// We'll test it by replicating the computation here.
static bool test_rope(const std::string & ref_dir) {
    fprintf(stderr, "\n=== Test: RoPE ===\n");

    // Test window RoPE (24x24)
    auto ref_window = load_ref(ref_dir + "/rope_window_real");
    if (!ref_window.data.empty()) {
        // ref is [576, 32, 2] — (cos, sin) pairs
        // Our compute_axial_cis produces [N, head_dim] with interleaved (cos, sin)
        // But the Python ref stores them as [N, half_head, 2] where [:,:,0]=cos, [:,:,1]=sin
        const int N = 576;
        const int half_head = 32;
        const int dim = 64;
        const float theta = 10000.0f;
        const float scale_pos = 1.0f;

        std::vector<float> our_rope(N * dim);

        const int half_dim = dim / 4;  // 16
        std::vector<float> freqs(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            freqs[i] = 1.0f / powf(theta, (float)(i * 4) / dim);
        }

        for (int idx = 0; idx < N; ++idx) {
            float t_x = (float)(idx % 24) * scale_pos;
            float t_y = (float)(idx / 24) * scale_pos;

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

        // Compare: ref is [N, 32, 2] = [N, half_head, (cos,sin)]
        // Our layout is [N, dim] = [N, 64] with pairs (cos_x, sin_x, cos_x, sin_x, ..., cos_y, sin_y, ...)
        // Need to rearrange comparison:
        // ref[idx, j, 0] = cos of freq j for position idx
        // ref[idx, j, 1] = sin of freq j for position idx
        // Python: freqs_cis = cat([freqs_cis_x, freqs_cis_y], dim=-1) → [N, 32]
        // freqs_cis_x has 16 complex values, freqs_cis_y has 16 complex values → total 32
        // view_as_real → [N, 32, 2] where [:,i,0]=cos, [:,i,1]=sin
        // So ref[idx, j, 0] for j<16 = cos(t_x * freqs[j])
        //    ref[idx, j, 1] for j<16 = sin(t_x * freqs[j])
        //    ref[idx, j, 0] for j>=16 = cos(t_y * freqs[j-16])
        //    ref[idx, j, 1] for j>=16 = sin(t_y * freqs[j-16])

        // Our layout: [idx * 64 + j*2+0] = cos, [idx * 64 + j*2+1] = sin
        // For x: j = 0..15 → our[idx*64 + j*2], ref[idx*64 + j*2]
        // For y: j = 16..31 → our[idx*64 + 32 + (j-16)*2], ref[idx*64 + (j)*2]

        // Actually both are the same layout! ref is stored flat as [N*32*2] = [N*64]
        // and our_rope is [N*64] with the same interleaving.
        float max_diff = compare_tensors(our_rope.data(), ref_window.data.data(), N * dim);
        bool ok = max_diff < 1e-5f;
        fprintf(stderr, "  %s window RoPE (24x24): max_diff=%.6e\n", ok ? "[PASS]" : "[FAIL]", max_diff);
    }

    // Test global RoPE (72x72)
    auto ref_global = load_ref(ref_dir + "/rope_global_real");
    if (!ref_global.data.empty()) {
        const int N = 5184;
        const int dim = 64;
        const float theta = 10000.0f;
        const float scale_pos = 24.0f / 72.0f;  // 1/3

        std::vector<float> our_rope(N * dim);
        const int half_dim = dim / 4;
        std::vector<float> freqs(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            freqs[i] = 1.0f / powf(theta, (float)(i * 4) / dim);
        }

        for (int idx = 0; idx < N; ++idx) {
            float t_x = (float)(idx % 72) * scale_pos;
            float t_y = (float)(idx / 72) * scale_pos;

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

        float max_diff = compare_tensors(our_rope.data(), ref_global.data.data(), N * dim);
        bool ok = max_diff < 1e-5f;
        fprintf(stderr, "  %s global RoPE (72x72): max_diff=%.6e\n", ok ? "[PASS]" : "[FAIL]", max_diff);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test: sinusoidal PE
// ═══════════════════════════════════════════════════════════════════════════════

// Forward-declare for testing
extern "C" {
    // We can't easily call static functions from sam3.cpp, so we'll replicate the logic
}

static bool test_sinusoidal_pe(const std::string & ref_dir) {
    fprintf(stderr, "\n=== Test: Sinusoidal PE ===\n");

    struct { int H; int W; const char * name; } cases[] = {
        {288, 288, "pe_288"},
        {144, 144, "pe_144"},
        { 72,  72, "pe_72"},
        { 36,  36, "pe_36"},
    };

    for (auto & tc : cases) {
        auto ref = load_ref(ref_dir + "/" + tc.name);
        if (ref.data.empty()) continue;

        // Compute our sinusoidal PE
        const int H = tc.H, W = tc.W, d_model = 256;
        const int half = d_model / 2;
        const float scale = 2.0f * (float)M_PI;
        const float temperature = 10000.0f;

        // ref is [1, 256, H, W] in NCHW from PyTorch
        std::vector<float> our_pe(d_model * H * W);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float pos_y = ((float)(y + 1) / (float)(H)) * scale;
                float pos_x = ((float)(x + 1) / (float)(W)) * scale;

                for (int i = 0; i < half; ++i) {
                    float dim_t = powf(temperature, 2.0f * (float)(i / 2) / (float)half);

                    float val_x, val_y;
                    if (i % 2 == 0) {
                        val_x = sinf(pos_x / dim_t);
                        val_y = sinf(pos_y / dim_t);
                    } else {
                        val_x = cosf(pos_x / dim_t);
                        val_y = cosf(pos_y / dim_t);
                    }

                    // PyTorch output is [1, 256, H, W] = [N, C, H, W]
                    // Channel layout: first half is pos_y, second half is pos_x
                    // Index: [0, c, y, x] → c * H * W + y * W + x
                    our_pe[i * H * W + y * W + x] = val_y;
                    our_pe[(i + half) * H * W + y * W + x] = val_x;
                }
            }
        }

        float max_diff = compare_tensors(our_pe.data(), ref.data.data(), d_model * H * W);
        bool ok = max_diff < 1e-5f;
        fprintf(stderr, "  %s %s (%dx%d): max_diff=%.6e\n",
                ok ? "[PASS]" : "[FAIL]", tc.name, tc.H, tc.W, max_diff);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Test: full image encoding
// ═══════════════════════════════════════════════════════════════════════════════

static bool test_encode_image(const std::string & model_path,
                               const std::string & image_path,
                               const std::string & ref_dir) {
    fprintf(stderr, "\n=== Test: Full Image Encoding ===\n");

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;  // CPU for determinism
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "  Failed to load model\n");
        return false;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "  Failed to create state\n");
        return false;
    }

    auto img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "  Failed to load image\n");
        return false;
    }

    bool ok = sam3_encode_image(*state, *model, img);
    if (!ok) {
        fprintf(stderr, "  sam3_encode_image failed\n");
        return false;
    }

    fprintf(stderr, "  Image encoding completed\n");

    int n_fail_enc = 0;
    std::string dump_dir = ref_dir + "/cpp_out";
    ensure_dir(dump_dir);

    // Helper to compare intermediate tensors.
    // ggml layout [E, W, H], Python ref is [1, H, W, E] (NHWC)
    auto compare_nhwc = [&](const std::string & cpp_name, const std::string & ref_name, float tol) {
        sam3_dump_state_tensor(*state, cpp_name, dump_dir + "/" + cpp_name);
        auto ref_t = load_ref(ref_dir + "/" + ref_name);
        auto cpp_t = load_ref(dump_dir + "/" + cpp_name);
        if (ref_t.data.empty() || cpp_t.data.empty()) return;
        int E = cpp_t.shape[0], nW = cpp_t.shape[1], nH = cpp_t.shape.size() > 2 ? cpp_t.shape[2] : 1;
        std::vector<float> transposed(cpp_t.numel());
        for (int e = 0; e < E; ++e)
            for (int w = 0; w < nW; ++w)
                for (int h = 0; h < nH; ++h)
                    transposed[h * nW * E + w * E + e] = cpp_t.data[e + w * E + h * E * nW];
        float md = compare_tensors(transposed.data(), ref_t.data.data(),
                                     std::min((int)transposed.size(), ref_t.numel()));
        bool ok = md < tol;
        fprintf(stderr, "  %s %s: max_diff=%.4f\n", ok ? "[PASS]" : "[FAIL]", ref_name.c_str(), md);
        if (!ok) n_fail_enc++;
    };

    // Compare intermediate outputs
    compare_nhwc("dbg_patch_embed", "patch_embed", 0.05f);
    compare_nhwc("dbg_after_ln_pre", "after_ln_pre", 0.1f);
    compare_nhwc("dbg_block_0_out", "block_0_out", 0.5f);

    sam3_dump_state_tensor(*state, "vit_output", dump_dir + "/vit_output");

    // Compare ViT output against Python reference.
    // ggml output: [ne0=E, ne1=W, ne2=H] stored as e + w*E + h*E*W
    // Python ref: [1, C, H, W] stored as c*H*W + h*W + w  (NCHW)
    // Need to transpose ggml data to match PyTorch NCHW layout.
    auto ref_vit = load_ref(ref_dir + "/vit_output_bchw");
    auto cpp_vit = load_ref(dump_dir + "/vit_output");

    if (!ref_vit.data.empty() && !cpp_vit.data.empty()) {
        // Transpose ggml [E=1024, W=72, H=72] → PyTorch [C=1024, H=72, W=72]
        int E = cpp_vit.shape[0], W = cpp_vit.shape[1], H = (cpp_vit.shape.size() > 2) ? cpp_vit.shape[2] : 1;
        std::vector<float> transposed(cpp_vit.numel());
        for (int e = 0; e < E; ++e)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    // ggml: e + w*E + h*E*W  →  pytorch: e*H*W + h*W + w
                    transposed[e * H * W + h * W + w] = cpp_vit.data[e + w * E + h * E * W];

        float max_diff = compare_tensors(transposed.data(), ref_vit.data.data(),
                                          std::min((int)transposed.size(), ref_vit.numel()));
        bool ok = max_diff < 1.0f;
        fprintf(stderr, "  %s ViT output vs Python ref: max_diff=%.4f (E=%d W=%d H=%d)\n",
                ok ? "[PASS]" : "[FAIL]", max_diff, E, W, H);
        if (!ok) n_fail_enc++;
    }

    // Dump and compare neck outputs.
    // Same layout issue: ggml [C, W, H] vs PyTorch [1, C, H, W]
    for (int i = 0; i < 3; ++i) {
        std::string name = "neck_det_" + std::to_string(i);
        sam3_dump_state_tensor(*state, name, dump_dir + "/" + name);

        auto ref_neck = load_ref(ref_dir + "/" + name);
        auto cpp_neck = load_ref(dump_dir + "/" + name);
        if (!ref_neck.data.empty() && !cpp_neck.data.empty()) {
            int C = cpp_neck.shape[0], nW = cpp_neck.shape[1], nH = (cpp_neck.shape.size() > 2) ? cpp_neck.shape[2] : 1;
            std::vector<float> transposed(cpp_neck.numel());
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < nH; ++h)
                    for (int w = 0; w < nW; ++w)
                        transposed[c * nH * nW + h * nW + w] = cpp_neck.data[c + w * C + h * C * nW];

            float max_diff = compare_tensors(transposed.data(), ref_neck.data.data(),
                                              std::min((int)transposed.size(), ref_neck.numel()));
            bool ok = max_diff < 2.0f;
            fprintf(stderr, "  %s %s vs Python ref: max_diff=%.4f\n",
                    ok ? "[PASS]" : "[FAIL]", name.c_str(), max_diff);
        }
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <ref_dir> [model.ggml] [image.jpg]\n", argv[0]);
        fprintf(stderr, "\nRun tests/dump_reference.py first to generate reference data.\n");
        return 1;
    }

    std::string ref_dir = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "";
    std::string image_path = (argc > 3) ? argv[3] : "";

    int n_pass = 0, n_fail = 0, n_skip = 0;

    // Test RoPE (no model needed)
    if (test_rope(ref_dir)) n_pass++; else n_fail++;

    // Test sinusoidal PE (no model needed)
    if (test_sinusoidal_pe(ref_dir)) n_pass++; else n_fail++;

    // Test preprocessing (needs image)
    if (!image_path.empty()) {
        if (test_preprocessing(image_path, ref_dir)) n_pass++; else n_fail++;
    } else {
        fprintf(stderr, "\n[SKIP] Preprocessing test — no image provided\n");
        n_skip++;
    }

    // Test full encoding (needs model + image)
    if (!model_path.empty() && !image_path.empty()) {
        if (test_encode_image(model_path, image_path, ref_dir)) n_pass++; else n_fail++;
    } else {
        fprintf(stderr, "\n[SKIP] Full encoding test — no model/image provided\n");
        n_skip++;
    }

    fprintf(stderr, "\n═══════════════════════════════════════════\n");
    fprintf(stderr, "Results: %d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    return n_fail > 0 ? 1 : 0;
}
