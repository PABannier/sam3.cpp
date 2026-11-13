// sam3-server: HTTP server for SAM3/SAM2 inference (llama-server style).
//
// Endpoints:
//   GET  /health                — status
//   GET  /models                — list *.ggml in models dir
//   POST /model/load            — switch model at runtime ({"model": "sam3-f16"})
//   POST /segment               — PVS (points/box), image base64 or path
//   POST /detect                — PCS (text prompt), SAM3 only
//   POST /track/new             — create tracking session
//   POST /track/frame           — process next frame
//   POST /track/instances       — add instance (PVS prompts)
//   POST /track/reset           — reset session
//   POST /video/upload          — multipart video → video_id (ffmpeg probe)
//   POST /video/open            — open http(s)/file/rtsp source → video_id
//   GET  /video/info            — video metadata
//   GET  /video/frame           — decoded frame as PNG
//   POST /video/track           — start a background tracking job (PVS instances[]/points
//                                 or text_prompt PCS re-detection); returns job_id
//   GET  /job/{id}              — job status/result
//   POST /job/{id}/cancel       — cancel a running job
//   GET  /                      — web UI
//
// Build: cmake -DSAM3_BUILD_EXAMPLES=ON (see examples/CMakeLists.txt)

#include "sam3.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

// stb_image implementation lives in libsam3 (sam3.cpp); only declarations here.
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

// ── CLI ───────────────────────────────────────────────────────────────────

struct server_params {
    std::string model_path;
    std::string models_dir = "models";
    std::string host       = "127.0.0.1";
    int         port       = 8080;
    int         n_threads  = 4;
    bool        use_gpu    = true;
    int         gpu_device = 0;
    int         encode_img_size = 0;
    int         n_slots    = 1;
    bool        webui      = true;
    float       score_threshold = 0.5f;
    float       nms_threshold   = 0.1f;
    int         mask_min_area        = 64;
    int         mask_fill_hole_area  = 2048;
    int         session_ttl_sec = 300;
    int         video_ttl_sec   = 600;
    int         job_ttl_sec     = 600;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -m, --model <path>       model file (or model name inside --models-dir)\n"
        "      --models-dir <dir>   directory with .ggml models (default: models)\n"
        "      --host <addr>        bind address (default: 127.0.0.1)\n"
        "      --port <port>        port (default: 8080)\n"
        "  -t, --threads <n>        CPU threads (default: 4)\n"
        "      --gpu                use GPU backend (default)\n"
        "      --nogpu              force CPU backend\n"
        "      --gpu-device <n>     GPU device index (default: 0)\n"
        "      --encode-img-size <n>  override encoder input resolution\n"
        "      --slots <n>          parallel inference slots (default: 1)\n"
        "      --no-webui           disable web UI\n"
        "      --score-threshold <f>  default detection score threshold (default: 0.5)\n"
        "      --nms-threshold <f>    default NMS IoU threshold (default: 0.1)\n"
        "      --mask-min-area <n>    cleanup: remove fg components <= n px (default: 64, 0 = off)\n"
        "      --mask-fill-hole-area <n>  cleanup: fill enclosed holes <= n px (default: 2048, 0 = off)\n"
        "      --session-ttl <sec>   evict idle tracker sessions after N s (default: 300)\n"
        "      --video-ttl <sec>     evict idle uploaded videos after N s (default: 600)\n"
        "      --job-ttl <sec>       drop finished job results after N s (default: 600)\n"
        "  -h, --help               show this help\n",
        argv0);
}

static bool parse_args(int argc, char ** argv, server_params & sp) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model")              sp.model_path = next(arg.c_str());
        else if (arg == "--models-dir")                   sp.models_dir = next(arg.c_str());
        else if (arg == "--host")                         sp.host = next(arg.c_str());
        else if (arg == "--port")                         sp.port = atoi(next(arg.c_str()).c_str());
        else if (arg == "-t" || arg == "--threads")       sp.n_threads = atoi(next(arg.c_str()).c_str());
        else if (arg == "--gpu")                          sp.use_gpu = true;
        else if (arg == "--nogpu")                        sp.use_gpu = false;
        else if (arg == "--gpu-device")                   sp.gpu_device = atoi(next(arg.c_str()).c_str());
        else if (arg == "--encode-img-size")              sp.encode_img_size = atoi(next(arg.c_str()).c_str());
        else if (arg == "--slots")                        sp.n_slots = atoi(next(arg.c_str()).c_str());
        else if (arg == "--no-webui")                     sp.webui = false;
        else if (arg == "--score-threshold")              sp.score_threshold = atof(next(arg.c_str()).c_str());
        else if (arg == "--nms-threshold")                sp.nms_threshold = atof(next(arg.c_str()).c_str());
        else if (arg == "--mask-min-area")                sp.mask_min_area = atoi(next(arg.c_str()).c_str());
        else if (arg == "--mask-fill-hole-area")          sp.mask_fill_hole_area = atoi(next(arg.c_str()).c_str());
        else if (arg == "--session-ttl")                  sp.session_ttl_sec = atoi(next(arg.c_str()).c_str());
        else if (arg == "--video-ttl")                    sp.video_ttl_sec = atoi(next(arg.c_str()).c_str());
        else if (arg == "--job-ttl")                      sp.job_ttl_sec = atoi(next(arg.c_str()).c_str());
        else if (arg == "-h" || arg == "--help")          { print_usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (sp.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ── base64 ────────────────────────────────────────────────────────────────

static const char b64_chars[] =
     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

static std::string base64_encode(const uint8_t * data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i + 1 < len) n |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) n |= (uint32_t)data[i + 2];
        out += b64_chars[(n >> 18) & 63];
        out += b64_chars[(n >> 12) & 63];
        out += b64_chars[i + 1 < len ? (n >> 6) & 63 : 64];
        out += b64_chars[i + 2 < len ? n & 63 : 64];
    }
    return out;
}

static std::vector<uint8_t> base64_decode(const std::string & s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    int acc = 0, bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int v = val(c);
        if (v < 0) continue;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// ── image helpers ─────────────────────────────────────────────────────────

static bool load_image_from_json(const json & req, sam3_image & img,
                                 std::string & err) {
    if (req.contains("image_path") && req["image_path"].is_string()) {
        img = sam3_load_image(req["image_path"].get<std::string>());
        if (img.data.empty()) {
            err = "failed to load image from path";
            return false;
        }
        return true;
    }
    if (req.contains("image_base64") && req["image_base64"].is_string()) {
        auto raw = base64_decode(req["image_base64"].get<std::string>());
        if (raw.empty()) {
            err = "failed to decode image_base64";
            return false;
        }
        int w = 0, h = 0, c = 0;
        uint8_t * px = stbi_load_from_memory(raw.data(), (int)raw.size(),
                                             &w, &h, &c, 3);
        if (!px) {
            err = "unsupported image format (jpeg/png expected)";
            return false;
        }
        img.width = w;
        img.height = h;
        img.channels = 3;
        img.data.assign(px, px + (size_t)w * h * 3);
        stbi_image_free(px);
        return true;
    }
    err = "missing image_path or image_base64";
    return false;
}

static void fill_pvs_params(const json & req, sam3_pvs_params & p) {
    if (req.contains("points") && req["points"].is_array()) {
        for (const auto & pt : req["points"]) {
            sam3_point pnt{pt.value("x", 0.0f), pt.value("y", 0.0f)};
            if (pt.value("label", 1) >= 0) p.pos_points.push_back(pnt);
            else                            p.neg_points.push_back(pnt);
        }
    }
    if (req.contains("box") && req["box"].is_object()) {
        const auto & b = req["box"];
        p.box = {b.value("x0", 0.0f), b.value("y0", 0.0f),
                 b.value("x1", 0.0f), b.value("y1", 0.0f)};
        p.use_box = true;
    }
    if (req.contains("multimask")) p.multimask = req["multimask"].get<bool>();
}

static json mask_to_json(const sam3_mask & m) {
    return json{
        {"width",      m.width},
        {"height",     m.height},
        {"iou_score",  m.iou_score},
        {"obj_score",  m.obj_score},
        {"instance_id", m.instance_id},
        {"mask_base64", base64_encode(m.data.data(), m.data.size())},
    };
}

static json detection_to_json(const sam3_detection & d) {
    json j = {
        {"box",   {{"x0", d.box.x0}, {"y0", d.box.y0},
                   {"x1", d.box.x1}, {"y1", d.box.y1}}},
        {"score", d.score},
        {"iou_score", d.iou_score},
        {"instance_id", d.instance_id},
    };
    j["mask"] = mask_to_json(d.mask);
    return j;
}

static json detections_to_json(const sam3_result & res) {
    json dets = json::array();
    for (const auto & d : res.detections) {
        dets.push_back(detection_to_json(d));
    }
    return dets;
}

// Optional mask post-processing ("cleanup": true in the request body):
// fills enclosed holes <= fill_hole_area and removes fg components <= min_area.
// Thresholds come from the body when present, otherwise from CLI defaults.
static void apply_cleanup(const json & body, const server_params & sp, sam3_result & result) {
    if (!body.value("cleanup", false)) return;
    const int min_area = body.contains("min_area")
            ? body["min_area"].get<int>() : sp.mask_min_area;
    const int fill_hole = body.contains("fill_hole_area")
            ? body["fill_hole_area"].get<int>() : sp.mask_fill_hole_area;
    for (auto & d : result.detections) {
        sam3_postprocess_mask(d.mask, min_area, fill_hole);
    }
}

// ── slots ─────────────────────────────────────────────────────────────────

struct slot {
    sam3_state_ptr state;
    std::mutex     mtx;
};

struct tracker_session {
    sam3_tracker_ptr tracker;
    sam3_state_ptr   state;
    std::mutex       mtx;
    std::atomic<int64_t> last_access_sec{0};
};

struct video_session {
    std::string     path;
    sam3_video_info info;
    std::mutex      mtx;
    std::atomic<int64_t> last_access_sec{0};
};

// Background /video/track job.
struct job {
    std::string id;
    std::string video_id;
    std::string video_path;
    sam3_video_info info;
    std::vector<sam3_pvs_params> inst_params;
    int            n_frames = 0;
    std::string    body;              // original request body (re-parsed by worker)
    std::atomic<bool> cancel{false};
    std::atomic<int>  frames_done{0};
    std::mutex    mtx;
    std::string   status = "running"; // running | done | cancelled | error
    std::string   error;
    json          result;
    int64_t       finished_at_sec = 0;
};

// RAII: releases a busy slot (busy_slots--) and unlocks its mutex on scope exit.
struct slot_lock {
    slot * s = nullptr;
    std::unique_lock<std::mutex> lock;
    std::atomic<int> * busy = nullptr;
    ~slot_lock() {
        if (busy) (*busy)--;
    }
};

static void png_write_cb(void * ctx, void * data, int size) {
    fwrite(data, 1, (size_t) size, (FILE *) ctx);
}

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    server_params sp;
    if (!parse_args(argc, argv, sp)) return 1;

    if (sp.model_path.find('/') == std::string::npos &&
        sp.model_path.find(".ggml") == std::string::npos) {
        sp.model_path = sp.models_dir + "/" + sp.model_path + ".ggml";
    }

    sam3_params params;
    params.model_path      = sp.model_path;
    params.n_threads       = sp.n_threads;
    params.use_gpu         = sp.use_gpu;
    params.gpu_device      = sp.gpu_device;
    params.encode_img_size = sp.encode_img_size;

    fprintf(stderr, "sam3-server: loading model %s\n", sp.model_path.c_str());
    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "sam3-server: failed to load model\n");
        return 1;
    }
    bool visual_only = sam3_is_visual_only(*model);
    fprintf(stderr, "sam3-server: model loaded (%s, %s)\n",
            visual_only ? "visual-only" : "SAM3",
            sp.use_gpu ? "GPU" : "CPU");

    std::vector<std::unique_ptr<slot>> slots;
    slots.reserve(sp.n_slots);
    for (int i = 0; i < sp.n_slots; ++i) {
        auto s = std::make_unique<slot>();
        s->state = sam3_create_state(*model, params);
        if (!s->state) {
            fprintf(stderr, "sam3-server: failed to create state slot %d\n", i);
            return 1;
        }
        slots.push_back(std::move(s));
    }

    std::mutex sessions_mtx;
    std::unordered_map<std::string, std::shared_ptr<tracker_session>> sessions;
    std::atomic<uint64_t> session_counter{0};

    std::mutex videos_mtx;
    std::unordered_map<std::string, std::shared_ptr<video_session>> videos;
    std::atomic<uint64_t> video_counter{0};

    std::mutex jobs_mtx;
    std::unordered_map<std::string, std::shared_ptr<job>> jobs;
    std::atomic<uint64_t> job_counter{0};
    std::atomic<int> active_jobs{0};

    std::atomic<bool> model_switching{false};
    std::atomic<int> busy_slots{0};

    httplib::Server svr;

    // ── /health ──
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        json j = {
            {"status", "ok"},
            {"model", sp.model_path},
            {"visual_only", visual_only},
            {"backend", sp.use_gpu ? "gpu" : "cpu"},
            {"slots", sp.n_slots},
            {"slots_busy", busy_slots.load()},
            {"active_jobs", active_jobs.load()},
        };
        res.set_content(j.dump(), "application/json");
    });

    // ── /models ──
    svr.Get("/models", [&](const httplib::Request &, httplib::Response & res) {
        std::vector<std::string> names;
#ifndef _WIN32
        DIR * dir = opendir(sp.models_dir.c_str());
        if (dir) {
            while (struct dirent * ent = readdir(dir)) {
                std::string name = ent->d_name;
                if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".ggml") == 0) {
                    names.push_back(sp.models_dir + "/" + name);
                }
            }
            closedir(dir);
        }
#endif
        std::sort(names.begin(), names.end());
        json models = json::array();
        for (const auto & n : names) models.push_back(n);
        res.set_content(json{{"models", models}}.dump(), "application/json");
    });

    // ── /model/load ──
    svr.Post("/model/load", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            if (!body.contains("model") || !body["model"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "missing model name"}}.dump(),
                                "application/json");
                return;
            }
            std::string path = body["model"].get<std::string>();
            if (path.find('/') == std::string::npos &&
                path.find(".ggml") == std::string::npos) {
                path = sp.models_dir + "/" + path + ".ggml";
            }

            // Reject concurrent switches; block new jobs until the swap finishes.
            if (model_switching.exchange(true)) {
                res.status = 503;
                res.set_content(json{{"error", "model switch already in progress"}}.dump(),
                                "application/json");
                return;
            }
            struct flag_reset {
                std::atomic<bool> * f;
                ~flag_reset() { f->store(false); }
            } switching_done{&model_switching};

            // Wait for running background jobs (they hold slots); then block
            // until all slots are free (no inference in flight).
            while (active_jobs.load() > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            std::vector<std::unique_lock<std::mutex>> locks;
            locks.reserve(slots.size());
            for (auto & s : slots) locks.emplace_back(s->mtx);

            sam3_params np = params;
            np.model_path = path;
            fprintf(stderr, "sam3-server: loading model %s\n", path.c_str());
            auto new_model = sam3_load_model(np);
            if (!new_model) {
                locks.clear();
                res.status = 500;
                res.set_content(json{{"error", "failed to load model: " + path}}.dump(),
                                "application/json");
                return;
            }

            std::vector<std::unique_ptr<slot>> new_slots;
            new_slots.reserve(sp.n_slots);
            for (int i = 0; i < sp.n_slots; ++i) {
                auto s = std::make_unique<slot>();
                s->state = sam3_create_state(*new_model, np);
                if (!s->state) {
                    locks.clear();
                    res.status = 500;
                    res.set_content(json{{"error", "failed to create state for new model"}}.dump(),
                                    "application/json");
                    return;
                }
                new_slots.push_back(std::move(s));
            }

            model = std::move(new_model);
            visual_only = sam3_is_visual_only(*model);
            std::vector<std::unique_ptr<slot>> old_slots = std::move(slots);
            slots = std::move(new_slots);
            {
                std::lock_guard<std::mutex> lock(sessions_mtx);
                sessions.clear();
            }
            locks.clear();
            old_slots.clear();
            sp.model_path = path;

            fprintf(stderr, "sam3-server: switched model to %s (%s, %s)\n",
                    path.c_str(),
                    visual_only ? "visual-only" : "SAM3",
                    sp.use_gpu ? "GPU" : "CPU");
            res.set_content(json{
                {"model", path},
                {"visual_only", visual_only},
                {"backend", sp.use_gpu ? "gpu" : "cpu"},
            }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── slot acquisition helper ──
    // Waits for a free slot (simple round-robin try-lock); returns nullptr if
    // the optional cancel flag is set while waiting. The returned slot_lock
    // releases the slot (busy_slots--) and unlocks the mutex on destruction.
    auto acquire_slot_lock = [&](const std::atomic<bool> * cancel)
            -> std::unique_ptr<slot_lock> {
        while (true) {
            if (cancel && cancel->load()) return nullptr;
            for (auto & s : slots) {
                std::unique_lock<std::mutex> lk(s->mtx, std::try_to_lock);
                if (lk.owns_lock()) {
                    busy_slots++;
                    auto out = std::make_unique<slot_lock>();
                    out->s    = s.get();
                    out->lock = std::move(lk);
                    out->busy = &busy_slots;
                    return out;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    };

    // ── /segment ──
    svr.Post("/segment", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            sam3_image img;
            std::string err;
            if (!load_image_from_json(body, img, err)) {
                res.status = 400;
                res.set_content(json{{"error", err}}.dump(), "application/json");
                return;
            }
            sam3_pvs_params p;
            fill_pvs_params(body, p);

            auto sl = acquire_slot_lock(nullptr);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!sam3_encode_image(*sl->s->state, *model, img)) {
                res.status = 500;
                res.set_content(json{{"error", "image encoding failed"}}.dump(),
                                "application/json");
                return;
            }
            auto result = sam3_segment_pvs(*sl->s->state, *model, p);
            auto t1 = std::chrono::high_resolution_clock::now();

            apply_cleanup(body, sp, result);

            json masks = json::array();
            for (const auto & d : result.detections) masks.push_back(mask_to_json(d.mask));
            json j = {
                {"masks", masks},
                {"time_ms", std::chrono::duration<double, std::milli>(t1 - t0).count()},
            };
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /detect ──
    svr.Post("/detect", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            if (visual_only) {
                res.status = 400;
                res.set_content(json{{"error", "text detection requires a SAM3 model"}}.dump(),
                                "application/json");
                return;
            }
            json body = json::parse(req.body);
            sam3_image img;
            std::string err;
            if (!load_image_from_json(body, img, err)) {
                res.status = 400;
                res.set_content(json{{"error", err}}.dump(), "application/json");
                return;
            }
            sam3_pcs_params p;
            if (!body.contains("text_prompt") && !body.contains("exemplars")) {
                res.status = 400;
                res.set_content(json{{"error", "missing text_prompt or exemplars"}}.dump(),
                                "application/json");
                return;
            }
            if (body.contains("text_prompt"))
                p.text_prompt = body["text_prompt"].get<std::string>();
            if (body.contains("exemplars")) {
                for (const auto & e : body["exemplars"]) {
                    if (!e.contains("box")) continue;
                    sam3_box b;
                    b.x0 = e["box"].value("x0", 0.0f);
                    b.y0 = e["box"].value("y0", 0.0f);
                    b.x1 = e["box"].value("x1", 0.0f);
                    b.y1 = e["box"].value("y1", 0.0f);
                    bool neg = e.value("label", "pos") == "neg";
                    if (neg) p.neg_exemplars.push_back(b);
                    else     p.pos_exemplars.push_back(b);
                }
            }
            p.score_threshold = body.contains("score_threshold")
                    ? body["score_threshold"].get<float>() : sp.score_threshold;
            p.nms_threshold   = body.contains("nms_threshold")
                    ? body["nms_threshold"].get<float>()   : sp.nms_threshold;

            auto sl = acquire_slot_lock(nullptr);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!sam3_encode_image(*sl->s->state, *model, img)) {
                res.status = 500;
                res.set_content(json{{"error", "image encoding failed"}}.dump(),
                                "application/json");
                return;
            }
            auto result = sam3_segment_pcs(*sl->s->state, *model, p);
            auto t1 = std::chrono::high_resolution_clock::now();

            apply_cleanup(body, sp, result);

            json j = {
                {"detections", detections_to_json(result)},
                {"time_ms", std::chrono::duration<double, std::milli>(t1 - t0).count()},
            };
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /track/new ──
    svr.Post("/track/new", [&](const httplib::Request &, httplib::Response & res) {
        auto sess = std::make_shared<tracker_session>();
        if (visual_only) {
            sam3_visual_track_params vp;
            sess->tracker = sam3_create_visual_tracker(*model, vp);
        } else {
            sam3_video_params vp;
            sess->tracker = sam3_create_tracker(*model, vp);
        }
        if (!sess->tracker) {
            res.status = 500;
            res.set_content(json{{"error", "failed to create tracker"}}.dump(),
                            "application/json");
            return;
        }
        sess->state = sam3_create_state(*model, params);
        if (!sess->state) {
            res.status = 500;
            res.set_content(json{{"error", "failed to create state"}}.dump(),
                            "application/json");
            return;
        }
        std::string id = "t" + std::to_string(++session_counter);
        sess->last_access_sec.store((int64_t)time(nullptr));
        {
            std::lock_guard<std::mutex> lock(sessions_mtx);
            sessions[id] = sess;
        }
        res.set_content(json{{"id", id}}.dump(), "application/json");
    });

    // ── /track/frame ──
    svr.Post("/track/frame", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            if (!body.contains("id")) {
                res.status = 400;
                res.set_content(json{{"error", "missing id"}}.dump(), "application/json");
                return;
            }
            std::shared_ptr<tracker_session> sess;
            {
                std::lock_guard<std::mutex> lock(sessions_mtx);
                auto it = sessions.find(body["id"].get<std::string>());
                if (it == sessions.end()) {
                    res.status = 404;
                    res.set_content(json{{"error", "unknown session"}}.dump(),
                                    "application/json");
                    return;
                }
                sess = it->second;
            }
            sess->last_access_sec.store((int64_t)time(nullptr));
            sam3_image img;
            std::string err;
            if (!load_image_from_json(body, img, err)) {
                res.status = 400;
                res.set_content(json{{"error", err}}.dump(), "application/json");
                return;
            }

            std::lock_guard<std::mutex> lock(sess->mtx);
            auto t0 = std::chrono::high_resolution_clock::now();
            sam3_result result;
            if (visual_only) {
                result = sam3_propagate_frame(*sess->tracker, *sess->state, *model, img);
            } else {
                result = sam3_track_frame(*sess->tracker, *sess->state, *model, img);
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            json j = {
                {"frame_index", sam3_tracker_frame_index(*sess->tracker)},
                {"detections", detections_to_json(result)},
                {"time_ms", std::chrono::duration<double, std::milli>(t1 - t0).count()},
            };
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /track/instances ──
    svr.Post("/track/instances", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            if (!body.contains("id")) {
                res.status = 400;
                res.set_content(json{{"error", "missing id"}}.dump(), "application/json");
                return;
            }
            std::shared_ptr<tracker_session> sess;
            {
                std::lock_guard<std::mutex> lock(sessions_mtx);
                auto it = sessions.find(body["id"].get<std::string>());
                if (it == sessions.end()) {
                    res.status = 404;
                    res.set_content(json{{"error", "unknown session"}}.dump(),
                                    "application/json");
                    return;
                }
                sess = it->second;
            }
            sess->last_access_sec.store((int64_t)time(nullptr));
            // instance must be added on the current (already encoded) frame
            sam3_image img;
            std::string err;
            if (!load_image_from_json(body, img, err)) {
                res.status = 400;
                res.set_content(json{{"error", err}}.dump(), "application/json");
                return;
            }
            sam3_pvs_params p;
            fill_pvs_params(body, p);

            std::lock_guard<std::mutex> lock(sess->mtx);
            if (!sam3_encode_image(*sess->state, *model, img)) {
                res.status = 500;
                res.set_content(json{{"error", "image encoding failed"}}.dump(),
                                "application/json");
                return;
            }
            int id = sam3_tracker_add_instance(*sess->tracker, *sess->state, *model, p);
            if (id < 0) {
                res.status = 500;
                res.set_content(json{{"error", "failed to add instance"}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(json{{"instance_id", id}}.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /track/reset ──
    svr.Post("/track/reset", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::shared_ptr<tracker_session> sess;
            {
                std::lock_guard<std::mutex> lock(sessions_mtx);
                auto it = sessions.find(body.value("id", ""));
                if (it != sessions.end()) {
                    sess = it->second;
                    sessions.erase(it);
                }
            }
            if (sess) {
                std::lock_guard<std::mutex> lock(sess->mtx);
                sam3_tracker_reset(*sess->tracker);
            }
            res.set_content(json{{"ok", true}}.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /video/upload ──
    svr.Post("/video/upload", [&](const httplib::Request & req, httplib::Response & res) {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(json{{"error", "missing multipart field 'file'"}}.dump(),
                            "application/json");
            return;
        }
        auto file = req.get_file_value("file");
        if (file.content.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty file"}}.dump(), "application/json");
            return;
        }
        const std::string base = "/tmp/sam3-server-videos";
        mkdir(base.c_str(), 0755);

        std::string id   = "v" + std::to_string(++video_counter);
        std::string name = file.filename;
        auto pos = name.find_last_of("/\\");
        if (pos != std::string::npos) name = name.substr(pos + 1);
        if (name.empty()) name = "video.bin";

        auto sess = std::make_shared<video_session>();
        sess->path = base + "/" + id + "_" + name;
        {
            FILE * fh = fopen(sess->path.c_str(), "wb");
            if (!fh) {
                res.status = 500;
                res.set_content(json{{"error", "failed to write upload"}}.dump(),
                                "application/json");
                return;
            }
            fwrite(file.content.data(), 1, file.content.size(), fh);
            fclose(fh);
        }
        sess->info = sam3_get_video_info(sess->path);
        if (sess->info.n_frames <= 0) {
            unlink(sess->path.c_str());
            res.status = 400;
            res.set_content(json{{"error", "failed to probe video (ffmpeg required)"}}.dump(),
                            "application/json");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(videos_mtx);
            sess->last_access_sec.store((int64_t)time(nullptr));
            videos[id] = sess;
        }
        res.set_content(json{
            {"video_id", id},
            {"width",    sess->info.width},
            {"height",   sess->info.height},
            {"n_frames", sess->info.n_frames},
            {"fps",      sess->info.fps},
        }.dump(), "application/json");
    });

    // ── /video/open ──
    svr.Post("/video/open", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::string url = body.value("url", "");
            if (url.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "missing 'url'"}}.dump(), "application/json");
                return;
            }
            const std::string base = "/tmp/sam3-server-videos";
            mkdir(base.c_str(), 0755);
            std::string id = "v" + std::to_string(++video_counter);
            auto sess = std::make_shared<video_session>();
            bool downloaded = false;
            bool live       = false;

            bool is_http   = url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0;
            bool is_stream = !is_http && (url.rfind("rtsp://", 0) == 0 ||
                                          url.rfind("rtmp://", 0) == 0 ||
                                          url.rfind("udp://", 0) == 0);
            if (is_http) {
                std::string host, path;
                int port = 0;
                size_t scheme = url.find("://");
                std::string rest = url.substr(scheme + 3);
                size_t slash = rest.find('/');
                std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
                path = slash == std::string::npos ? "/" : rest.substr(slash);
                port = (url.rfind("https://", 0) == 0) ? 443 : 80;
                size_t colon = hostport.rfind(':');
                if (colon != std::string::npos) {
                    port = atoi(hostport.substr(colon + 1).c_str());
                    host = hostport.substr(0, colon);
                } else {
                    host = hostport;
                }
                if (host.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "invalid url"}}.dump(), "application/json");
                    return;
                }
                std::string name = "download.bin";
                size_t last = path.find_last_of('/');
                if (last != std::string::npos && last + 1 < path.size())
                    name = path.substr(last + 1);
                if (name.empty()) name = "download.bin";
                sess->path = base + "/" + id + "_" + name;
                FILE * fh = fopen(sess->path.c_str(), "wb");
                if (!fh) {
                    res.status = 500;
                    res.set_content(json{{"error", "failed to open download file"}}.dump(),
                                    "application/json");
                    return;
                }
                httplib::Client cli(host, port);
                cli.set_follow_location(true);
                auto r = cli.Get(path, [&](const char * data, size_t len) {
                    fwrite(data, 1, len, fh);
                    return true;
                });
                fclose(fh);
                if (!r || r->status != 200) {
                    unlink(sess->path.c_str());
                    res.status = 502;
                    res.set_content(json{{"error", "download failed"}}.dump(),
                                    "application/json");
                    return;
                }
                downloaded = true;
            } else if (is_stream) {
                sess->path = url;  // ffmpeg reads streams directly
                live = true;
            } else {
                struct stat st;
                if (stat(url.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
                    res.status = 404;
                    res.set_content(json{{"error", "file not found (or not a regular file)"}}.dump(),
                                    "application/json");
                    return;
                }
                sess->path = url;
            }

            sess->info = sam3_get_video_info(sess->path);
            if (sess->info.width <= 0 || sess->info.height <= 0) {
                if (downloaded) unlink(sess->path.c_str());
                res.status = 400;
                res.set_content(json{{"error", "failed to probe video (ffmpeg/ffprobe required)"}}.dump(),
                                "application/json");
                return;
            }
            if (live) sess->info.n_frames = 0;  // unknown length

            {
                std::lock_guard<std::mutex> lock(videos_mtx);
                sess->last_access_sec.store((int64_t)time(nullptr));
                videos[id] = sess;
            }
            res.set_content(json{
                {"video_id", id},
                {"width",    sess->info.width},
                {"height",   sess->info.height},
                {"n_frames", sess->info.n_frames},
                {"fps",      sess->info.fps},
                {"live",     sess->info.n_frames <= 0},
            }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /video/info ──
    svr.Get("/video/info", [&](const httplib::Request & req, httplib::Response & res) {
        std::string id = req.get_param_value("video_id");
        std::shared_ptr<video_session> sess;
        {
            std::lock_guard<std::mutex> lock(videos_mtx);
            auto it = videos.find(id);
            if (it == videos.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown video_id"}}.dump(),
                                "application/json");
                return;
            }
            sess = it->second;
        }
        sess->last_access_sec.store((int64_t)time(nullptr));
        res.set_content(json{
            {"video_id", id},
            {"width",    sess->info.width},
            {"height",   sess->info.height},
            {"n_frames", sess->info.n_frames},
            {"fps",      sess->info.fps},
        }.dump(), "application/json");
    });

    // ── /video/frame ──
    svr.Get("/video/frame", [&](const httplib::Request & req, httplib::Response & res) {
        std::string id    = req.get_param_value("video_id");
        int         frame = atoi(req.get_param_value("frame").c_str());
        std::shared_ptr<video_session> sess;
        {
            std::lock_guard<std::mutex> lock(videos_mtx);
            auto it = videos.find(id);
            if (it == videos.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown video_id"}}.dump(),
                                "application/json");
                return;
            }
            sess = it->second;
        }
        sess->last_access_sec.store((int64_t)time(nullptr));
        std::lock_guard<std::mutex> lock(sess->mtx);
        sam3_image img = sam3_decode_video_frame(sess->path, frame);
        if (img.data.empty()) {
            res.status = 404;
            res.set_content(json{{"error", "failed to decode frame"}}.dump(),
                            "application/json");
            return;
        }
        FILE * f = tmpfile();
        if (!f) {
            res.status = 500;
            res.set_content(json{{"error", "tmpfile failed"}}.dump(),
                            "application/json");
            return;
        }
        stbi_write_png_to_func(png_write_cb, f, img.width, img.height, 3,
                               img.data.data(), img.width * 3);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> png((size_t) size);
        fread(png.data(), 1, (size_t) size, f);
        fclose(f);
        res.set_content(png.data(), (size_t) size, "image/png");
    });

    // ── /video/track (background job) ──
    auto run_job = [&](std::shared_ptr<job> j) {
        struct vdec_guard {
            sam3_video_decoder * d = nullptr;
            ~vdec_guard() {
                if (d) sam3_video_decoder_close(d);
            }
        };
        try {
            json body = json::parse(j->body);
            {
                std::lock_guard<std::mutex> lock(videos_mtx);
                auto it = videos.find(j->video_id);
                if (it != videos.end()) {
                    it->second->last_access_sec.store((int64_t)time(nullptr));
                }
            }
            auto sl = acquire_slot_lock(&j->cancel);
            if (!sl) {
                std::lock_guard<std::mutex> lock(j->mtx);
                j->status = "cancelled";
                j->finished_at_sec = (int64_t)time(nullptr);
                active_jobs--;
                return;
            }
            auto t0 = std::chrono::high_resolution_clock::now();

            vdec_guard vg;
            vg.d = sam3_video_decoder_open(j->video_path, j->info.width, j->info.height);
            if (!vg.d) throw std::runtime_error("failed to open video decoder");
            sam3_image f0;
            if (!sam3_video_decoder_read(vg.d, f0)) throw std::runtime_error("failed to decode frame 0");
            if (!sam3_encode_image(*sl->s->state, *model, f0)) throw std::runtime_error("image encoding failed");

            sam3_tracker_ptr tracker;
            if (visual_only) {
                sam3_visual_track_params vtp;
                vtp.max_keep_alive    = body.value("max_keep_alive", 100);
                vtp.recondition_every = body.value("recondition_every", 16);
                tracker = sam3_create_visual_tracker(*model, vtp);
            } else {
                sam3_video_params vp;
                vp.text_prompt          = body.value("text_prompt", "");
                vp.score_threshold      = body.contains("score_threshold")
                        ? body["score_threshold"].get<float>() : sp.score_threshold;
                vp.nms_threshold        = body.contains("nms_threshold")
                        ? body["nms_threshold"].get<float>()   : sp.nms_threshold;
                vp.assoc_iou_threshold  = body.value("assoc_iou_threshold", 0.1f);
                vp.hotstart_delay       = body.value("hotstart_delay", 0);
                vp.max_keep_alive       = body.value("max_keep_alive", 100);
                vp.recondition_every    = body.value("recondition_every", 16);
                vp.fill_hole_area       = body.value("fill_hole_area", 16);
                tracker = sam3_create_tracker(*model, vp);
            }
            if (!tracker) throw std::runtime_error("failed to create tracker");

            std::vector<int> inst_ids;
            for (const auto & ip : j->inst_params) {
                int inst_id = sam3_tracker_add_instance(*tracker, *sl->s->state, *model, ip);
                if (inst_id < 0) throw std::runtime_error("failed to add instance");
                inst_ids.push_back(inst_id);
            }

            json frames = json::array();
            json dets0 = json::array();
            for (size_t k = 0; k < j->inst_params.size(); ++k) {
                sam3_result res0 = sam3_segment_pvs(*sl->s->state, *model, j->inst_params[k]);
                if (res0.detections.empty()) continue;
                json d = detection_to_json(res0.detections[0]);
                d["instance_id"] = inst_ids[k];
                dets0.push_back(d);
            }
            frames.push_back(json{{"index", 0}, {"detections", dets0}});
            j->frames_done.store(1);

            bool cancelled = false;
            for (int f = 1; f < j->n_frames; ++f) {
                if (j->cancel.load()) { cancelled = true; break; }
                sam3_image img;
                if (!sam3_video_decoder_read(vg.d, img)) break;
                sam3_result result;
                if (visual_only) {
                    result = sam3_propagate_frame(*tracker, *sl->s->state, *model, img);
                } else {
                    result = sam3_track_frame(*tracker, *sl->s->state, *model, img);
                }
                fprintf(stderr, "[video] %s frame %d/%d: %zu detections\n",
                        j->video_id.c_str(), f, j->n_frames, result.detections.size());
                frames.push_back(json{
                    {"index", f},
                    {"detections", detections_to_json(result)},
                });
                j->frames_done.store(f + 1);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(j->mtx);
            j->status = cancelled ? "cancelled" : "done";
            j->result = json{
                {"video_id", j->video_id},
                {"frames", frames},
                {"time_ms", std::chrono::duration<double, std::milli>(t1 - t0).count()},
            };
            j->finished_at_sec = (int64_t)time(nullptr);
        } catch (const std::exception & e) {
            std::lock_guard<std::mutex> lock(j->mtx);
            j->status = "error";
            j->error  = e.what();
            j->finished_at_sec = (int64_t)time(nullptr);
        }
        active_jobs--;
    };

    svr.Post("/video/track", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::string id = body.value("video_id", "");
            std::shared_ptr<video_session> sess;
            {
                std::lock_guard<std::mutex> lock(videos_mtx);
                auto it = videos.find(id);
                if (it == videos.end()) {
                    res.status = 404;
                    res.set_content(json{{"error", "unknown video_id"}}.dump(),
                                    "application/json");
                    return;
                }
                sess = it->second;
                sess->last_access_sec.store((int64_t)time(nullptr));
            }
            std::vector<sam3_pvs_params> inst_params;
            if (body.contains("instances") && body["instances"].is_array() &&
                !body["instances"].empty()) {
                for (const auto & inst : body["instances"]) {
                    sam3_pvs_params ip;
                    fill_pvs_params(inst, ip);
                    if (ip.pos_points.empty() && ip.neg_points.empty() && !ip.use_box) {
                        res.status = 400;
                        res.set_content(json{{"error", "empty instance prompt"}}.dump(),
                                        "application/json");
                        return;
                    }
                    inst_params.push_back(ip);
                }
            } else {
                sam3_pvs_params p;
                fill_pvs_params(body, p);
                if (p.pos_points.empty() && p.neg_points.empty() && !p.use_box &&
                    !body.contains("text_prompt")) {
                    res.status = 400;
                    res.set_content(json{{"error", "no prompts provided"}}.dump(),
                                    "application/json");
                    return;
                }
                inst_params.push_back(p);
            }

            int n = sess->info.n_frames;
            int max_frames = body.value("max_frames", 0);
            if (n <= 0) {
                if (max_frames <= 0) {
                    res.status = 400;
                    res.set_content(json{{"error", "live stream requires max_frames"}}.dump(),
                                    "application/json");
                    return;
                }
                n = max_frames;
            } else if (max_frames > 0 && max_frames < n) {
                n = max_frames;
            }

            if (model_switching.load()) {
                res.status = 503;
                res.set_content(json{{"error", "model switch in progress, retry shortly"}}.dump(),
                                "application/json");
                return;
            }

            auto j = std::make_shared<job>();
            j->id          = "j" + std::to_string(++job_counter);
            j->video_id    = id;
            j->video_path  = sess->path;
            j->info        = sess->info;
            j->inst_params = std::move(inst_params);
            j->n_frames    = n;
            j->body        = req.body;
            {
                std::lock_guard<std::mutex> lock(jobs_mtx);
                jobs[j->id] = j;
            }
            active_jobs++;
            std::thread([&run_job, j] { run_job(j); }).detach();

            res.set_content(json{
                {"job_id",  j->id},
                {"status",  "running"},
                {"n_frames", n},
            }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── /job/{id} ──
    svr.Get(R"(/job/([^/]+))", [&](const httplib::Request & req, httplib::Response & res) {
        std::shared_ptr<job> j;
        {
            std::lock_guard<std::mutex> lock(jobs_mtx);
            auto it = jobs.find(req.matches[1]);
            if (it == jobs.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown job_id"}}.dump(),
                                "application/json");
                return;
            }
            j = it->second;
        }
        std::lock_guard<std::mutex> lock(j->mtx);
        json out = {
            {"job_id",  j->id},
            {"status",  j->status},
            {"progress", {
                {"frames_done", j->frames_done.load()},
                {"n_frames",    j->n_frames},
            }},
        };
        if (j->status == "done" || j->status == "cancelled") out["result"] = j->result;
        if (j->status == "error") out["error"] = j->error;
        res.set_content(out.dump(), "application/json");
    });

    // ── /job/{id}/cancel ──
    svr.Post(R"(/job/([^/]+)/cancel)", [&](const httplib::Request & req, httplib::Response & res) {
        std::shared_ptr<job> j;
        {
            std::lock_guard<std::mutex> lock(jobs_mtx);
            auto it = jobs.find(req.matches[1]);
            if (it == jobs.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown job_id"}}.dump(),
                                "application/json");
                return;
            }
            j = it->second;
        }
        j->cancel.store(true);
        res.set_content(json{{"job_id", j->id}, {"status", "cancelling"}}.dump(),
                        "application/json");
    });

    // ── web UI ──
    if (sp.webui) {
        #include "webui_html.h"
        svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
            res.set_content(sam3_server_index_html, "text/html");
        });
    }

    // ── TTL janitor ──
    std::atomic<bool> stop_janitor{false};
    std::thread janitor([&] {
        while (!stop_janitor.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            int64_t now = (int64_t)time(nullptr);
            {
                std::lock_guard<std::mutex> lock(sessions_mtx);
                for (auto it = sessions.begin(); it != sessions.end();) {
                    if (now - it->second->last_access_sec.load() > sp.session_ttl_sec) {
                        it = sessions.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(videos_mtx);
                for (auto it = videos.begin(); it != videos.end();) {
                    if (now - it->second->last_access_sec.load() > sp.video_ttl_sec) {
                        const std::string & p = it->second->path;
                        if (p.rfind("/tmp/sam3-server-videos", 0) == 0) {
                            unlink(p.c_str());
                        }
                        it = videos.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(jobs_mtx);
                for (auto it = jobs.begin(); it != jobs.end();) {
                    if (it->second->finished_at_sec != 0 &&
                        now - it->second->finished_at_sec > sp.job_ttl_sec) {
                        it = jobs.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    });

    fprintf(stderr, "sam3-server: listening on http://%s:%d\n",
            sp.host.c_str(), sp.port);
    svr.listen(sp.host.c_str(), sp.port);
    stop_janitor.store(true);
    janitor.join();
    return 0;
}
