#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

struct ref_tensor_f32 {
    std::vector<float> data;
    std::vector<int> shape;

    int numel() const {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }
};

struct ref_tensor_i32 {
    std::vector<int32_t> data;
    std::vector<int> shape;

    int numel() const {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }
};

struct compare_result {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float cosine_sim = 0.0f;
    int n_bad = 0;
    int n_total = 0;
};

static inline std::vector<int> load_shape_file(const std::string & path) {
    std::vector<int> shape;
    std::ifstream f(path);
    if (!f) {
        return shape;
    }

    std::string line;
    std::getline(f, line);
    size_t pos = 0;
    while (pos < line.size()) {
        size_t end = line.find(',', pos);
        if (end == std::string::npos) {
            end = line.size();
        }
        shape.push_back(std::stoi(line.substr(pos, end - pos)));
        pos = end + 1;
    }
    return shape;
}

static inline ref_tensor_f32 load_ref_f32(const std::string & path) {
    ref_tensor_f32 t;
    t.shape = load_shape_file(path + ".shape");
    if (t.shape.empty()) {
        return t;
    }

    std::ifstream f(path + ".bin", std::ios::binary);
    if (!f) {
        t.shape.clear();
        return t;
    }

    t.data.resize(t.numel());
    f.read(reinterpret_cast<char *>(t.data.data()), t.numel() * sizeof(float));
    return t;
}

static inline ref_tensor_i32 load_ref_i32(const std::string & path) {
    ref_tensor_i32 t;
    t.shape = load_shape_file(path + ".shape");
    if (t.shape.empty()) {
        return t;
    }

    std::ifstream f(path + ".bin", std::ios::binary);
    if (!f) {
        t.shape.clear();
        return t;
    }

    t.data.resize(t.numel());
    f.read(reinterpret_cast<char *>(t.data.data()), t.numel() * sizeof(int32_t));
    return t;
}

static inline compare_result compare_tensors(const float * a,
                                             const float * b,
                                             int n,
                                             float atol = 1e-4f) {
    compare_result r;
    r.n_total = n;

    double sum_diff = 0.0;
    double dot_ab = 0.0;
    double dot_aa = 0.0;
    double dot_bb = 0.0;

    for (int i = 0; i < n; ++i) {
        float diff = 0.0f;
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            diff = (std::isinf(a[i]) && std::isinf(b[i]) &&
                    ((a[i] > 0.0f) == (b[i] > 0.0f))) ? 0.0f : INFINITY;
        } else {
            diff = fabsf(a[i] - b[i]);
            dot_ab += (double) a[i] * (double) b[i];
            dot_aa += (double) a[i] * (double) a[i];
            dot_bb += (double) b[i] * (double) b[i];
        }

        if (diff > r.max_diff) {
            r.max_diff = diff;
        }
        if (diff > atol) {
            r.n_bad++;
        }
        sum_diff += diff;
    }

    r.mean_diff = n > 0 ? (float) (sum_diff / n) : 0.0f;
    const double denom = sqrt(dot_aa) * sqrt(dot_bb);
    r.cosine_sim = denom > 0.0 ? (float) (dot_ab / denom) : 0.0f;
    return r;
}

static inline int compare_exact_i32(const std::vector<int32_t> & got,
                                    const std::vector<int32_t> & ref) {
    const size_t n = got.size() < ref.size() ? got.size() : ref.size();
    int n_bad = got.size() == ref.size() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != ref[i]) {
            ++n_bad;
        }
    }
    return n_bad;
}

static inline bool ensure_dir(const std::string & path) {
    if (path.empty()) {
        return true;
    }

    std::string cur;
    size_t pos = 0;
    if (path[0] == '/') {
        cur = "/";
        pos = 1;
    }

    while (pos <= path.size()) {
        size_t next = path.find('/', pos);
        if (next == std::string::npos) {
            next = path.size();
        }

        const std::string part = path.substr(pos, next - pos);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') {
                cur.push_back('/');
            }
            cur += part;
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }

        pos = next + 1;
    }

    return true;
}
