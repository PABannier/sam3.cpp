/**
 * sam3_benchmark — exhaustive latency benchmark for all model variants.
 *
 * Tracks an object (point prompt) across N video frames on CPU and Metal,
 * then prints a formatted comparison table.
 *
 * Each model × backend run is isolated in a forked subprocess so that a
 * crash (e.g. unsupported Metal op) does not kill the entire benchmark.
 *
 * Usage:
 *   sam3_benchmark [options]
 *
 * Options:
 *   --models-dir <path>   Directory with .ggml files  (default: models/)
 *   --video <path>        Video file                   (default: data/test_video.mp4)
 *   --point-x <f>         Click point X                (default: 315.0)
 *   --point-y <f>         Click point Y                (default: 250.0)
 *   --n-frames <n>        Frames to track              (default: 10)
 *   --n-threads <n>       CPU threads                  (default: 4)
 *   --cpu-only            Skip Metal runs
 *   --gpu-only            Skip CPU runs
 *   --filter <substr>     Only run models whose filename contains <substr>
 */

#include "sam3.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// ── Wire format for child→parent result ─────────────────────────────────────

struct BenchWire {
    double  t_load_ms;
    double  t_frame0_ms;
    double  t_track_avg_ms;
    double  t_total_ms;
    int     n_detections;
    int     ok;            // 1 = success, 0 = failure
    char    error[128];
};

// ── Display result ──────────────────────────────────────────────────────────

struct BenchResult {
    std::string model_name;
    std::string backend;
    int64_t     file_size      = 0;
    double      t_load_ms      = 0.0;
    double      t_frame0_ms    = 0.0;
    double      t_track_avg_ms = 0.0;
    double      t_total_ms     = 0.0;
    int         n_detections   = 0;
    bool        success        = false;
    std::string error;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string format_size(int64_t bytes) {
    char buf[32];
    if (bytes >= (int64_t)1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (int64_t)1024 * 1024) {
        snprintf(buf, sizeof(buf), "%lld MB", (long long)(bytes / (1024 * 1024)));
    } else {
        snprintf(buf, sizeof(buf), "%lld KB", (long long)(bytes / 1024));
    }
    return buf;
}

static std::string format_time_short(double ms) {
    if (ms < 0) return "  -";
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", ms);
    return buf;
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string strip_extension(const std::string & filename) {
    auto pos = filename.rfind('.');
    return (pos != std::string::npos) ? filename.substr(0, pos) : filename;
}

// Sort key: family → size → precision
static int model_sort_key(const std::string & name) {
    int family = 0;
    if (name.find("sam3-visual") != std::string::npos)          family = 1;
    else if (name.find("sam3") != std::string::npos)            family = 0;
    else if (name.find("sam2.1") != std::string::npos)          family = 3;
    else if (name.find("sam2") != std::string::npos)            family = 2;

    int size = 0;
    if (name.find("tiny") != std::string::npos)                 size = 0;
    else if (name.find("small") != std::string::npos)           size = 1;
    else if (name.find("base_plus") != std::string::npos)       size = 2;
    else if (name.find("large") != std::string::npos)           size = 3;

    int prec = 0;
    if (name.find("f32") != std::string::npos)                  prec = 0;
    else if (name.find("f16") != std::string::npos)             prec = 1;
    else if (name.find("q8_0") != std::string::npos)            prec = 2;
    else if (name.find("q4_1") != std::string::npos)            prec = 3;
    else if (name.find("q4_0") != std::string::npos)            prec = 4;

    return family * 10000 + size * 100 + prec;
}

// ── Model discovery ─────────────────────────────────────────────────────────

struct ModelEntry {
    std::string path;
    std::string name;
    int64_t     file_size;
};

static std::vector<ModelEntry> discover_models(const std::string & dir,
                                                const std::string & filter) {
    std::vector<ModelEntry> entries;
    DIR * d = opendir(dir.c_str());
    if (!d) {
        fprintf(stderr, "ERROR: cannot open models directory '%s'\n", dir.c_str());
        return entries;
    }
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string fname = ent->d_name;
        if (!ends_with(fname, ".ggml")) continue;
        if (!filter.empty() && fname.find(filter) == std::string::npos) continue;

        std::string full_path = dir + "/" + fname;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0) continue;

        ModelEntry e;
        e.path      = full_path;
        e.name      = strip_extension(fname);
        e.file_size = st.st_size;
        entries.push_back(e);
    }
    closedir(d);

    std::sort(entries.begin(), entries.end(), [](const ModelEntry & a, const ModelEntry & b) {
        return model_sort_key(a.name) < model_sort_key(b.name);
    });
    return entries;
}

// ── Child process: run a single benchmark ───────────────────────────────────

static void child_benchmark(const std::string & model_path,
                             bool use_gpu,
                             const std::string & video_path,
                             int n_frames,
                             float px, float py,
                             int n_threads,
                             int keyframe_interval,
                             int write_fd) {
    BenchWire wire = {};

    auto write_result = [&]() {
        (void)write(write_fd, &wire, sizeof(wire));
        close(write_fd);
    };

    auto fail = [&](const char * msg) {
        wire.ok = 0;
        snprintf(wire.error, sizeof(wire.error), "%s", msg);
        write_result();
        _exit(1);
    };

    // Decode frames
    std::vector<sam3_image> frames(n_frames);
    for (int f = 0; f < n_frames; f++) {
        frames[f] = sam3_decode_video_frame(video_path, f);
        if (frames[f].data.empty()) { fail("decode frame failed"); return; }
    }

    // Load model
    int64_t t0 = ggml_time_us();

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu    = use_gpu;
    params.n_threads  = n_threads;

    auto model = sam3_load_model(params);
    if (!model) { fail("load failed"); return; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fail("state failed"); return; }

    bool visual_only = sam3_is_visual_only(*model);
    sam3_tracker_ptr tracker;

    if (visual_only) {
        sam3_visual_track_params vtp;
        vtp.max_keep_alive     = 100;
        vtp.recondition_every  = 16;
        vtp.keyframe_interval  = keyframe_interval;
        tracker = sam3_create_visual_tracker(*model, vtp);
    } else {
        sam3_video_params vp;
        vp.hotstart_delay      = 0;
        vp.max_keep_alive      = 100;
        vp.keyframe_interval   = keyframe_interval;
        tracker = sam3_create_tracker(*model, vp);
    }
    if (!tracker) { fail("tracker failed"); return; }

    wire.t_load_ms = (ggml_time_us() - t0) / 1000.0;

    // Frame 0: encode + add instance
    t0 = ggml_time_us();

    if (!sam3_encode_image(*state, *model, frames[0])) {
        fail("encode f0 failed"); return;
    }

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;

    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    if (inst_id < 0) { fail("add_instance failed"); return; }

    wire.t_frame0_ms = (ggml_time_us() - t0) / 1000.0;

    // Frames 1..N-1: track / propagate
    double t_track_sum = 0.0;
    sam3_result last_result;

    for (int f = 1; f < n_frames; f++) {
        t0 = ggml_time_us();

        if (visual_only) {
            last_result = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
        } else {
            last_result = sam3_track_frame(*tracker, *state, *model, frames[f]);
        }

        double dt = (ggml_time_us() - t0) / 1000.0;
        t_track_sum += dt;

        fprintf(stderr, "    frame %d/%d  %.0f ms  (%zu det)\n",
                f, n_frames - 1, dt, last_result.detections.size());
    }

    if (n_frames > 1) {
        wire.t_track_avg_ms = t_track_sum / (n_frames - 1);
    }
    wire.t_total_ms   = wire.t_frame0_ms + t_track_sum;
    wire.n_detections = (int)last_result.detections.size();
    wire.ok           = 1;

    write_result();
    _exit(0);
}

// ── Parent: launch child and collect result ─────────────────────────────────

static BenchResult run_benchmark_isolated(const ModelEntry & entry,
                                           bool use_gpu,
                                           const std::string & video_path,
                                           int n_frames,
                                           float px, float py,
                                           int n_threads,
                                           int keyframe_interval) {
    BenchResult res;
    res.model_name = entry.name;
    res.backend    = use_gpu ? "Metal" : "CPU";
    res.file_size  = entry.file_size;

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        res.error = "pipe() failed";
        return res;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        res.error = "fork() failed";
        return res;
    }

    if (pid == 0) {
        // Child
        close(pipefd[0]);
        child_benchmark(entry.path, use_gpu, video_path,
                        n_frames, px, py, n_threads, keyframe_interval, pipefd[1]);
        _exit(1);
    }

    // Parent
    close(pipefd[1]);

    BenchWire wire = {};
    ssize_t nread = read(pipefd[0], &wire, sizeof(wire));
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    if (nread == (ssize_t)sizeof(wire) && wire.ok) {
        res.t_load_ms      = wire.t_load_ms;
        res.t_frame0_ms    = wire.t_frame0_ms;
        res.t_track_avg_ms = wire.t_track_avg_ms;
        res.t_total_ms     = wire.t_total_ms;
        res.n_detections   = wire.n_detections;
        res.success        = true;
    } else if (nread == (ssize_t)sizeof(wire) && !wire.ok) {
        res.error = wire.error;
    } else if (WIFSIGNALED(status)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "crashed (signal %d)", WTERMSIG(status));
        res.error = buf;
    } else {
        res.error = "child failed (no result)";
    }

    return res;
}

// ── Table printing ──────────────────────────────────────────────────────────

static void print_table(const std::vector<BenchResult> & results,
                         const std::string & video_path,
                         float px, float py, int n_frames, int n_threads,
                         int keyframe_interval) {
    printf("\n");
    printf("=========================================================================================================\n");
    printf("SAM3.CPP BENCHMARK  —  %d frames, point=(%.1f, %.1f), threads=%d, keyframe_interval=%d\n",
           n_frames, px, py, n_threads, keyframe_interval);
    printf("video: %s\n", video_path.c_str());
    printf("=========================================================================================================\n\n");

    printf("  %3s | %-36s | %7s | %6s | %9s | %9s | %13s | %10s | %3s | %s\n",
           "#", "Model", "Size", "Backend", "Load (ms)", "Init (ms)", "Track/fr (ms)", "Total (ms)", "Det", "Status");
    printf("------+--------------------------------------+---------+--------+-----------+-----------+---------------+------------+-----+--------\n");

    for (size_t i = 0; i < results.size(); i++) {
        const auto & r = results[i];
        if (r.success) {
            printf("  %3d | %-36s | %7s | %6s | %9s | %9s | %13s | %10s | %3d | OK\n",
                   (int)(i + 1),
                   r.model_name.c_str(),
                   format_size(r.file_size).c_str(),
                   r.backend.c_str(),
                   format_time_short(r.t_load_ms).c_str(),
                   format_time_short(r.t_frame0_ms).c_str(),
                   format_time_short(r.t_track_avg_ms).c_str(),
                   format_time_short(r.t_total_ms).c_str(),
                   r.n_detections);
        } else {
            printf("  %3d | %-36s | %7s | %6s | %9s | %9s | %13s | %10s | %3s | FAIL: %s\n",
                   (int)(i + 1),
                   r.model_name.c_str(),
                   format_size(r.file_size).c_str(),
                   r.backend.c_str(),
                   r.t_load_ms > 0 ? format_time_short(r.t_load_ms).c_str() : "-",
                   "-", "-", "-", "-",
                   r.error.c_str());
        }
    }

    printf("------+--------------------------------------+---------+--------+-----------+-----------+---------------+------------+-----+--------\n");

    int n_ok = 0, n_fail = 0;
    for (const auto & r : results) { if (r.success) n_ok++; else n_fail++; }
    printf("\nSUMMARY: %d runs, %d OK, %d FAIL\n", (int)results.size(), n_ok, n_fail);
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::string models_dir = "models/";
    std::string video_path = "data/test_video.mp4";
    float       px         = 315.0f;
    float       py         = 250.0f;
    int         n_frames   = 10;
    int         n_threads  = 4;
    int         keyframe_interval = 15;
    bool        cpu_only   = false;
    bool        gpu_only   = false;
    std::string filter;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--models-dir" && i + 1 < argc) { models_dir = argv[++i]; }
        else if (arg == "--video"      && i + 1 < argc) { video_path = argv[++i]; }
        else if (arg == "--point-x"    && i + 1 < argc) { px = (float)atof(argv[++i]); }
        else if (arg == "--point-y"    && i + 1 < argc) { py = (float)atof(argv[++i]); }
        else if (arg == "--n-frames"   && i + 1 < argc) { n_frames = atoi(argv[++i]); }
        else if (arg == "--n-threads"  && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (arg == "--cpu-only")  { cpu_only = true; }
        else if (arg == "--gpu-only")  { gpu_only = true; }
        else if (arg == "--filter"     && i + 1 < argc) { filter = argv[++i]; }
        else if (arg == "--keyframe-interval" && i + 1 < argc) { keyframe_interval = atoi(argv[++i]); }
        else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s [options]\n"
                "  --models-dir <path>       Models directory       (default: models/)\n"
                "  --video <path>            Video file             (default: data/test_video.mp4)\n"
                "  --point-x <f>             Click X                (default: 315.0)\n"
                "  --point-y <f>             Click Y                (default: 250.0)\n"
                "  --n-frames <n>            Frames to track        (default: 10)\n"
                "  --n-threads <n>           CPU threads            (default: 4)\n"
                "  --keyframe-interval <n>   Flow keyframe interval (default: 15, 0=disable)\n"
                "  --cpu-only                Skip Metal runs\n"
                "  --gpu-only                Skip CPU runs\n"
                "  --filter <substr>         Filter model filenames\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (n_frames < 2) {
        fprintf(stderr, "ERROR: --n-frames must be >= 2\n");
        return 1;
    }

    // Discover models
    auto entries = discover_models(models_dir, filter);
    if (entries.empty()) {
        fprintf(stderr, "ERROR: no .ggml files found in '%s'", models_dir.c_str());
        if (!filter.empty()) fprintf(stderr, " (filter: '%s')", filter.c_str());
        fprintf(stderr, "\n");
        return 1;
    }

    fprintf(stderr, "Found %zu model(s)\n", entries.size());
    for (const auto & e : entries) {
        fprintf(stderr, "  %s  (%s)\n", e.name.c_str(), format_size(e.file_size).c_str());
    }

    // Validate video
    auto vinfo = sam3_get_video_info(video_path);
    if (vinfo.n_frames <= 0) {
        fprintf(stderr, "ERROR: cannot read video '%s'\n", video_path.c_str());
        return 1;
    }
    if (n_frames > vinfo.n_frames) {
        fprintf(stderr, "WARNING: video has %d frames, clamping to %d\n", vinfo.n_frames, vinfo.n_frames);
        n_frames = vinfo.n_frames;
    }
    fprintf(stderr, "Video: %dx%d, %d frames, %.1f fps\n",
            vinfo.width, vinfo.height, vinfo.n_frames, vinfo.fps);

    // Build run list
    struct RunSpec {
        const ModelEntry * entry;
        bool use_gpu;
    };
    std::vector<RunSpec> runs;
    for (const auto & e : entries) {
        if (!gpu_only) runs.push_back({&e, false});
        if (!cpu_only) runs.push_back({&e, true});
    }

    std::sort(runs.begin(), runs.end(), [](const RunSpec & a, const RunSpec & b) {
        int ka = model_sort_key(a.entry->name);
        int kb = model_sort_key(b.entry->name);
        if (ka != kb) return ka < kb;
        return a.use_gpu > b.use_gpu;  // Metal first
    });

    fprintf(stderr, "\nStarting %zu benchmark runs (each in a subprocess)...\n\n", runs.size());

    // Run benchmarks
    int64_t t_wall_start = ggml_time_us();
    std::vector<BenchResult> results;
    results.reserve(runs.size());

    for (size_t i = 0; i < runs.size(); i++) {
        const auto & run = runs[i];
        const char * backend_str = run.use_gpu ? "Metal" : "CPU";

        fprintf(stderr, "[%3zu/%zu] %s (%s) ...\n",
                i + 1, runs.size(), run.entry->name.c_str(), backend_str);

        auto res = run_benchmark_isolated(*run.entry, run.use_gpu, video_path,
                                           n_frames, px, py, n_threads, keyframe_interval);
        results.push_back(res);

        if (res.success) {
            fprintf(stderr, "  -> OK  load=%.0fms  init=%.0fms  track/fr=%.0fms  total=%.0fms  det=%d\n\n",
                    res.t_load_ms, res.t_frame0_ms, res.t_track_avg_ms, res.t_total_ms, res.n_detections);
        } else {
            fprintf(stderr, "  -> FAIL: %s\n\n", res.error.c_str());
        }
    }

    double t_wall_ms = (ggml_time_us() - t_wall_start) / 1000.0;

    // Print table
    print_table(results, video_path, px, py, n_frames, n_threads, keyframe_interval);

    double t_wall_s = t_wall_ms / 1000.0;
    int mins = (int)(t_wall_s / 60.0);
    int secs = (int)(t_wall_s) % 60;
    printf("Total wall time: %dm %ds\n\n", mins, secs);

    return 0;
}
