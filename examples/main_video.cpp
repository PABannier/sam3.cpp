// sam3_video — Interactive video segmentation + tracking example
//
// Usage: sam3_video --model <path.ggml> --video <path>
//
// Controls:
//   Mode selector chooses how to initialize tracking instances:
//     Text:   Type a text prompt → auto-detect instances on each frame.
//     Box:    Drag a bounding box on a paused frame → add instance.
//     Points: Click positive/negative points → add instance.
//   In all modes, clicking on an existing tracked mask refines it.
//   [Play]/[Pause]/[Step] for playback control.
//   [Export] saves mask PNGs per frame.

#include "sam3.h"

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

#ifdef __APPLE__
#ifndef GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#endif
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────────

static const float INSTANCE_COLORS[][3] = {
    {1.0f, 0.2f, 0.2f}, {0.2f, 0.6f, 1.0f}, {0.2f, 0.9f, 0.3f},
    {1.0f, 0.8f, 0.1f}, {0.8f, 0.3f, 0.9f}, {1.0f, 0.5f, 0.1f},
    {0.1f, 0.9f, 0.9f}, {0.9f, 0.4f, 0.6f}, {0.5f, 0.8f, 0.2f},
    {0.3f, 0.3f, 1.0f}, {1.0f, 0.6f, 0.7f}, {0.6f, 1.0f, 0.5f},
};
static constexpr int N_COLORS = sizeof(INSTANCE_COLORS) / sizeof(INSTANCE_COLORS[0]);

enum vtrack_mode { VMODE_TEXT, VMODE_BOX, VMODE_POINTS };

struct vapp_state {
    // Model
    sam3_params             params;
    std::shared_ptr<sam3_model> model;
    sam3_state_ptr          state;
    sam3_tracker_ptr        tracker;

    // Video
    std::string             video_path;
    sam3_video_info         video_info;

    // Current frame
    sam3_image              frame;
    int                     frame_index = 0;
    GLuint                  tex_frame   = 0;
    bool                    frame_encoded = false;

    // Tracking
    char                    text_prompt[256] = {};
    sam3_video_params       track_params;
    sam3_result             result;
    bool                    tracker_created = false;

    // Interaction mode
    vtrack_mode             init_mode = VMODE_TEXT;

    // Manual instance creation (box/points on paused frame)
    std::vector<sam3_point> init_pos_points;
    std::vector<sam3_point> init_neg_points;
    sam3_box                init_box = {0, 0, 0, 0};
    bool                    has_init_box = false;
    bool                    dragging = false;
    float                   drag_x0 = 0, drag_y0 = 0;

    // Playback
    bool                    playing   = false;
    float                   play_speed = 1.0f;
    Uint32                  last_frame_time = 0;

    // Display
    bool                    show_masks = true;
    float                   canvas_x = 0, canvas_y = 0;
    float                   canvas_w = 0, canvas_h = 0;

    // Status
    char                    status[256] = "Ready.";
    bool                    busy = false;

    // Model type
    bool                    visual_only = false;
    sam3_visual_track_params visual_track_params;

    // Timeline: per-frame instance presence
    // timeline_instances[frame_index] = list of {instance_id, score}
    struct frame_entry {
        std::vector<std::pair<int, float>> instances; // (id, score)
    };
    std::vector<frame_entry> timeline;
    int                     timeline_max_frame = -1; // highest frame tracked so far
};

static GLuint upload_texture(const uint8_t* data, int w, int h, int ch, GLuint existing = 0) {
    GLuint tex = existing;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLenum fmt = (ch == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
    return tex;
}

static void build_frame_overlay(vapp_state& app, std::vector<uint8_t>& overlay) {
    if (app.frame.data.empty()) return;
    int w = app.frame.width;
    int h = app.frame.height;

    overlay.resize(w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        overlay[4*i+0] = app.frame.data[3*i+0];
        overlay[4*i+1] = app.frame.data[3*i+1];
        overlay[4*i+2] = app.frame.data[3*i+2];
        overlay[4*i+3] = 255;
    }

    if (app.show_masks) {
        for (size_t d = 0; d < app.result.detections.size(); ++d) {
            const auto& det = app.result.detections[d];
            if (det.mask.data.empty()) continue;
            int ci = det.instance_id > 0 ? (det.instance_id - 1) % N_COLORS : (int)(d % N_COLORS);
            const float* c = INSTANCE_COLORS[ci];
            float alpha = 0.4f;

            int mw = det.mask.width;
            int mh = det.mask.height;
            for (int y = 0; y < std::min(h, mh); ++y) {
                for (int x = 0; x < std::min(w, mw); ++x) {
                    if (det.mask.data[y * mw + x] > 127) {
                        int idx = (y * w + x) * 4;
                        overlay[idx+0] = (uint8_t)(overlay[idx+0]*(1-alpha) + c[0]*255*alpha);
                        overlay[idx+1] = (uint8_t)(overlay[idx+1]*(1-alpha) + c[1]*255*alpha);
                        overlay[idx+2] = (uint8_t)(overlay[idx+2]*(1-alpha) + c[2]*255*alpha);
                    }
                }
            }
        }
    }
}

static bool screen_to_image(const vapp_state& app, float sx, float sy,
                             float& ix, float& iy) {
    if (app.canvas_w <= 0 || app.canvas_h <= 0) return false;
    ix = (sx - app.canvas_x) / app.canvas_w * app.frame.width;
    iy = (sy - app.canvas_y) / app.canvas_h * app.frame.height;
    return ix >= 0 && iy >= 0 && ix < app.frame.width && iy < app.frame.height;
}

static void create_tracker(vapp_state& app) {
    if (app.visual_only) {
        // SAM2 / visual-only: use visual tracker (no text detection)
        app.visual_track_params.assoc_iou_threshold = app.track_params.assoc_iou_threshold;
        app.visual_track_params.max_keep_alive      = app.track_params.max_keep_alive;
        app.visual_track_params.recondition_every    = app.track_params.recondition_every;
        app.visual_track_params.fill_hole_area       = app.track_params.fill_hole_area;
        app.tracker = sam3_create_visual_tracker(*app.model, app.visual_track_params);
    } else {
        if (app.init_mode == VMODE_TEXT)
            app.track_params.text_prompt = app.text_prompt;
        else
            app.track_params.text_prompt = "";
        app.tracker = sam3_create_tracker(*app.model, app.track_params);
    }
    app.tracker_created = (app.tracker != nullptr);
    snprintf(app.status, sizeof(app.status), app.tracker_created
             ? "Tracker created. Press Play or add instances."
             : "Failed to create tracker.");
}

static void decode_and_track(vapp_state& app, int fi) {
    app.frame = sam3_decode_video_frame(app.video_path, fi);
    if (app.frame.data.empty()) {
        snprintf(app.status, sizeof(app.status), "Failed to decode frame %d.", fi);
        app.playing = false;
        return;
    }
    app.frame_index = fi;

    if (app.tracker_created) {
        if (app.visual_only)
            app.result = sam3_propagate_frame(*app.tracker, *app.state, *app.model, app.frame);
        else
            app.result = sam3_track_frame(*app.tracker, *app.state, *app.model, app.frame);
        app.frame_encoded = true;

        // Record instance presence in timeline
        if (fi >= (int)app.timeline.size())
            app.timeline.resize(fi + 1);
        app.timeline[fi].instances.clear();
        for (const auto& det : app.result.detections)
            app.timeline[fi].instances.push_back({det.instance_id, det.score});
        if (fi > app.timeline_max_frame) app.timeline_max_frame = fi;

        snprintf(app.status, sizeof(app.status), "Frame %d/%d — %d objects tracked",
                 fi, app.video_info.n_frames, (int)app.result.detections.size());
    } else {
        // Just encode and show the frame without tracking
        sam3_encode_image(*app.state, *app.model, app.frame);
        app.frame_encoded = true;
        app.result = {};
        snprintf(app.status, sizeof(app.status), "Frame %d/%d — no tracker active",
                 fi, app.video_info.n_frames);
    }
}

// Check if a click position lands on any tracked instance mask.
// Returns the instance_id or -1.
static int find_instance_at(const vapp_state& app, float ix, float iy) {
    int px = (int)ix, py = (int)iy;
    for (const auto& det : app.result.detections) {
        if (det.mask.data.empty()) continue;
        if (px >= 0 && px < det.mask.width &&
            py >= 0 && py < det.mask.height &&
            det.mask.data[py * det.mask.width + px] > 127) {
            return det.instance_id;
        }
    }
    return -1;
}

static void add_instance_from_prompts(vapp_state& app) {
    if (!app.tracker_created || !app.frame_encoded) return;
    sam3_pvs_params pvs;
    pvs.pos_points = app.init_pos_points;
    pvs.neg_points = app.init_neg_points;
    if (app.has_init_box) {
        pvs.box = app.init_box;
        pvs.use_box = true;
    }
    pvs.multimask = false;

    // sam3_tracker_add_instance runs PVS internally and registers the instance
    // in the tracker's memory bank.  We must NOT call decode_and_track afterwards
    // because sam3_track_frame increments tracker.frame_index, which would
    // desynchronize the tracker's internal counter from the actual frame.
    //
    // Instead, run PVS again (cheap — image is already encoded) to obtain a
    // display mask and append it to the current result.
    int new_id = sam3_tracker_add_instance(*app.tracker, *app.state, *app.model, pvs);
    if (new_id >= 0) {
        snprintf(app.status, sizeof(app.status), "Added instance #%d", new_id);

        // Re-run PVS to get the mask for display (image features are still valid)
        auto pvs_result = sam3_segment_pvs(*app.state, *app.model, pvs);
        if (!pvs_result.detections.empty()) {
            sam3_detection det = pvs_result.detections[0];
            det.instance_id = new_id;
            det.mask.instance_id = new_id;
            app.result.detections.push_back(std::move(det));
        }
        // Update timeline for current frame
        int fi = app.frame_index;
        if (fi >= (int)app.timeline.size())
            app.timeline.resize(fi + 1);
        app.timeline[fi].instances.clear();
        for (const auto& d : app.result.detections)
            app.timeline[fi].instances.push_back({d.instance_id, d.score});
        if (fi > app.timeline_max_frame) app.timeline_max_frame = fi;
    } else {
        snprintf(app.status, sizeof(app.status), "Failed to add instance");
    }

    // Clear pending prompts
    app.init_pos_points.clear();
    app.init_neg_points.clear();
    app.init_box = {0, 0, 0, 0};
    app.has_init_box = false;
}

static void clear_init_prompts(vapp_state& app) {
    app.init_pos_points.clear();
    app.init_neg_points.clear();
    app.init_box = {0, 0, 0, 0};
    app.has_init_box = false;
}

static void export_frame_masks(const vapp_state& app) {
    for (size_t i = 0; i < app.result.detections.size(); ++i) {
        char path[256];
        snprintf(path, sizeof(path), "frame%04d_mask%02d.png",
                 app.frame_index, (int)i);
        if (sam3_save_mask(app.result.detections[i].mask, path)) {
            fprintf(stderr, "Exported %s\n", path);
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    vapp_state app;
    app.params.n_threads = 4;
    app.params.use_gpu   = true;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            app.params.model_path = argv[++i];
        } else if (strcmp(argv[i], "--video") == 0 && i+1 < argc) {
            app.video_path = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) {
            app.params.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
            app.params.use_gpu = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model <path.ggml> --video <path>\n"
                "          [--threads N] [--no-gpu]\n", argv[0]);
            return 0;
        }
    }

    if (app.params.model_path.empty()) {
        fprintf(stderr, "Error: --model is required.\n");
        return 1;
    }

    // ── Init SDL2 + OpenGL ───────────────────────────────────────────────────

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

#ifdef __APPLE__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    const char* glsl_version = "#version 150";
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    const char* glsl_version = "#version 130";
#endif

    SDL_Window* window = SDL_CreateWindow(
        "sam3 — Video Tracking",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 800,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_ctx);
    SDL_GL_SetSwapInterval(1);

    // ── Init ImGui ───────────────────────────────────────────────────────────

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window, gl_ctx);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ── Load model ───────────────────────────────────────────────────────────

    fprintf(stderr, "Loading model: %s\n", app.params.model_path.c_str());
    app.model = sam3_load_model(app.params);
    if (!app.model) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    app.state = sam3_create_state(*app.model, app.params);
    if (!app.state) {
        fprintf(stderr, "Failed to create state.\n");
        return 1;
    }

    app.visual_only = sam3_is_visual_only(*app.model);
    if (app.visual_only) {
        auto mt = sam3_get_model_type(*app.model);
        fprintf(stderr, "Model: %s — text tracking disabled\n",
                mt == SAM3_MODEL_SAM2 ? "SAM2" : "SAM3 visual-only");
        app.init_mode = VMODE_BOX;
    }

    // ── Load video info ──────────────────────────────────────────────────────

    if (!app.video_path.empty()) {
        app.video_info = sam3_get_video_info(app.video_path);
        if (app.video_info.n_frames > 0) {
            // Auto-create tracker and encode first frame
            create_tracker(app);
            if (app.tracker_created) {
                decode_and_track(app, 0);
            } else {
                app.frame = sam3_decode_video_frame(app.video_path, 0);
                app.frame_index = 0;
            }
            snprintf(app.status, sizeof(app.status),
                     "Video: %dx%d, %d frames, %.1f fps. Pause and annotate to add instances.",
                     app.video_info.width, app.video_info.height,
                     app.video_info.n_frames, app.video_info.fps);
        } else {
            snprintf(app.status, sizeof(app.status), "Failed to read video info.");
        }
    } else {
        snprintf(app.status, sizeof(app.status), "No video specified. Use --video <path>.");
    }

    // ── Overlay buffer ───────────────────────────────────────────────────────

    std::vector<uint8_t> overlay_buf;

    // ── Main loop ────────────────────────────────────────────────────────────

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) running = false;

            // Keyboard shortcuts
            if (event.type == SDL_KEYDOWN && !io.WantCaptureKeyboard) {
                if (event.key.keysym.sym == SDLK_SPACE) {
                    app.playing = !app.playing;
                    if (app.playing) app.last_frame_time = SDL_GetTicks();
                } else if (event.key.keysym.sym == SDLK_RIGHT) {
                    if (app.frame_index + 1 < app.video_info.n_frames) {
                        decode_and_track(app, app.frame_index + 1);
                    }
                } else if (event.key.keysym.sym == SDLK_LEFT) {
                    if (app.frame_index > 0) {
                        app.frame = sam3_decode_video_frame(app.video_path, app.frame_index - 1);
                        app.frame_index--;
                        // The state's encoded features are now stale (they belong
                        // to the old frame).  Clear frame_encoded so that
                        // interactions like box-draw or refine won't operate on
                        // the wrong features.  Also clear displayed results since
                        // they correspond to the old frame.
                        app.frame_encoded = false;
                        app.result = {};
                    }
                }
            }
        }

        // ── Auto-advance when playing ────────────────────────────────────────

        if (app.playing && app.video_info.fps > 0) {
            Uint32 now = SDL_GetTicks();
            float interval_ms = 1000.0f / (app.video_info.fps * app.play_speed);
            if (now - app.last_frame_time >= (Uint32)interval_ms) {
                int next = app.frame_index + 1;
                if (next < app.video_info.n_frames) {
                    decode_and_track(app, next);
                    app.last_frame_time = now;
                } else {
                    app.playing = false;
                    snprintf(app.status, sizeof(app.status), "End of video.");
                }
            }
        }

        // ── ImGui frame ──────────────────────────────────────────────────────

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)win_w, (float)win_h));
        ImGui::Begin("sam3_video", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        // ── Top bar ──────────────────────────────────────────────────────────

        // Mode selector
        ImGui::Text("Mode:");
        ImGui::SameLine();
        if (!app.visual_only) {
            if (ImGui::RadioButton("Text", app.init_mode == VMODE_TEXT)) {
                app.init_mode = VMODE_TEXT;
                clear_init_prompts(app);
            }
            ImGui::SameLine();
        }
        if (ImGui::RadioButton("Box", app.init_mode == VMODE_BOX)) {
            app.init_mode = VMODE_BOX;
            clear_init_prompts(app);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Points", app.init_mode == VMODE_POINTS)) {
            app.init_mode = VMODE_POINTS;
            clear_init_prompts(app);
        }

        // Text prompt (only for VMODE_TEXT on full SAM3 models)
        if (app.init_mode == VMODE_TEXT && !app.visual_only) {
            ImGui::SameLine();
            ImGui::Text("  Text:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200);
            ImGui::InputText("##prompt", app.text_prompt, sizeof(app.text_prompt));
        }

        // Playback buttons
        ImGui::SameLine();
        if (app.playing) {
            if (ImGui::Button("Pause")) app.playing = false;
        } else {
            if (ImGui::Button("Play")) {
                app.playing = true;
                app.last_frame_time = SDL_GetTicks();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Step >>") && app.tracker_created) {
            if (app.frame_index + 1 < app.video_info.n_frames) {
                decode_and_track(app, app.frame_index + 1);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            app.playing = false;
            app.tracker_created = false;
            app.frame_encoded = false;
            app.tracker.reset();
            app.result = {};
            app.frame_index = 0;
            app.timeline.clear();
            app.timeline_max_frame = -1;
            clear_init_prompts(app);
            if (app.visual_only && app.init_mode == VMODE_TEXT)
                app.init_mode = VMODE_BOX;
            // Re-create tracker and encode first frame
            create_tracker(app);
            if (app.tracker_created && !app.video_path.empty()) {
                decode_and_track(app, 0);
            } else if (!app.video_path.empty()) {
                app.frame = sam3_decode_video_frame(app.video_path, 0);
            }
            snprintf(app.status, sizeof(app.status), "Reset. Ready to annotate.");
        }

        ImGui::Text("Frame: %d/%d   FPS: %.1f   Objects: %d   Speed: %.1fx",
                     app.frame_index, app.video_info.n_frames,
                     app.video_info.fps, (int)app.result.detections.size(),
                     app.play_speed);

        // ── Video canvas ─────────────────────────────────────────────────────

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float panel_h = 120.0f;
        float timeline_h = 52.0f;
        float canvas_max_h = avail.y - panel_h - timeline_h;
        float canvas_max_w = avail.x;

        if (!app.frame.data.empty()) {
            build_frame_overlay(app, overlay_buf);
            app.tex_frame = upload_texture(overlay_buf.data(),
                                            app.frame.width, app.frame.height,
                                            4, app.tex_frame);

            float iw = (float)app.frame.width;
            float ih = (float)app.frame.height;
            float scale = std::min(canvas_max_w / iw, canvas_max_h / ih);
            float dw = iw * scale;
            float dh = ih * scale;

            ImVec2 pos = ImGui::GetCursorScreenPos();
            float offset_x = (canvas_max_w - dw) * 0.5f;
            pos.x += offset_x;

            app.canvas_x = pos.x;
            app.canvas_y = pos.y;
            app.canvas_w = dw;
            app.canvas_h = dh;

            ImGui::SetCursorScreenPos(pos);
            // Use InvisibleButton so ImGui doesn't capture mouse over the canvas
            ImGui::InvisibleButton("canvas", ImVec2(dw, dh));
            bool canvas_hovered = ImGui::IsItemHovered();

            // ── Mouse interaction on canvas (ImGui API, not SDL) ────────────
            if (canvas_hovered) {
                float mx = io.MousePos.x, my = io.MousePos.y;
                float ix, iy;

                // Left click → start drag (box mode) or add positive point
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        int hit_id = find_instance_at(app, ix, iy);
                        if (hit_id >= 0 && app.tracker_created && app.frame_encoded) {
                            std::vector<sam3_point> p = {{ix, iy}};
                            std::vector<sam3_point> neg;
                            bool ok = sam3_refine_instance(*app.tracker, *app.state, *app.model,
                                                           hit_id, p, neg);
                            snprintf(app.status, sizeof(app.status),
                                     ok ? "Refined instance #%d with positive point"
                                        : "Failed to refine instance #%d", hit_id);
                        } else if (app.init_mode == VMODE_BOX) {
                            app.dragging = true;
                            app.drag_x0 = ix;
                            app.drag_y0 = iy;
                        } else if (app.init_mode == VMODE_POINTS) {
                            app.init_pos_points.push_back({ix, iy});
                            if (app.tracker_created && app.frame_encoded) {
                                add_instance_from_prompts(app);
                            }
                        }
                    }
                }

                // Right click → negative point or refine with negative
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        int hit_id = find_instance_at(app, ix, iy);
                        if (hit_id >= 0 && app.tracker_created && app.frame_encoded) {
                            std::vector<sam3_point> p;
                            for (const auto& det : app.result.detections) {
                                if (det.instance_id != hit_id || det.mask.data.empty()) continue;
                                float cx = 0, cy = 0; int n = 0;
                                int mw = det.mask.width;
                                for (int pi = 0; pi < (int)det.mask.data.size(); ++pi) {
                                    if (det.mask.data[pi] > 127) {
                                        cx += static_cast<float>(pi % mw);
                                        cy += static_cast<float>(pi / mw);  // NOLINT: integer row index
                                        ++n;
                                    }
                                }
                                if (n > 0) p.push_back({cx / n, cy / n});
                                break;
                            }
                            std::vector<sam3_point> neg = {{ix, iy}};
                            bool ok = sam3_refine_instance(*app.tracker, *app.state, *app.model,
                                                           hit_id, p, neg);
                            snprintf(app.status, sizeof(app.status),
                                     ok ? "Refined instance #%d with negative point"
                                        : "Failed to refine instance #%d", hit_id);
                        } else if (app.init_mode == VMODE_POINTS) {
                            app.init_neg_points.push_back({ix, iy});
                        }
                    }
                }
            }

            // Left button release → finish box drag
            if (app.dragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                float mx = io.MousePos.x, my = io.MousePos.y;
                float ix, iy;
                if (screen_to_image(app, mx, my, ix, iy)) {
                    float dx = ix - app.drag_x0;
                    float dy = iy - app.drag_y0;
                    if (dx*dx + dy*dy > 25.0f) {
                        app.init_box.x0 = std::min(app.drag_x0, ix);
                        app.init_box.y0 = std::min(app.drag_y0, iy);
                        app.init_box.x1 = std::max(app.drag_x0, ix);
                        app.init_box.y1 = std::max(app.drag_y0, iy);
                        app.has_init_box = true;
                        if (app.tracker_created && app.frame_encoded) {
                            add_instance_from_prompts(app);
                        }
                    }
                }
                app.dragging = false;
            }

            // Draw the frame via DrawList (not ImGui::Image which eats mouse)
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddImage((ImTextureID)(intptr_t)app.tex_frame,
                         ImVec2(pos.x, pos.y),
                         ImVec2(pos.x + dw, pos.y + dh));
            for (size_t d = 0; d < app.result.detections.size(); ++d) {
                const auto& det = app.result.detections[d];
                int ci = det.instance_id > 0 ? (det.instance_id - 1) % N_COLORS : (int)(d % N_COLORS);
                const float* c = INSTANCE_COLORS[ci];
                ImU32 col = IM_COL32((int)(c[0]*255), (int)(c[1]*255),
                                     (int)(c[2]*255), 200);

                float sx0 = app.canvas_x + det.box.x0 / iw * dw;
                float sy0 = app.canvas_y + det.box.y0 / ih * dh;
                float sx1 = app.canvas_x + det.box.x1 / iw * dw;
                float sy1 = app.canvas_y + det.box.y1 / ih * dh;
                dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1), col, 0, 0, 2);

                char label[64];
                snprintf(label, sizeof(label), "#%d %.2f", det.instance_id, det.score);
                dl->AddText(ImVec2(sx0 + 2, sy0 - 14), col, label);
            }

            // Draw pending init points (green = positive, red = negative)
            for (const auto& p : app.init_pos_points) {
                float sx = app.canvas_x + p.x / iw * dw;
                float sy = app.canvas_y + p.y / ih * dh;
                dl->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(0, 255, 0, 220));
                dl->AddCircle(ImVec2(sx, sy), 6, IM_COL32(255, 255, 255, 255), 0, 2);
            }
            for (const auto& p : app.init_neg_points) {
                float sx = app.canvas_x + p.x / iw * dw;
                float sy = app.canvas_y + p.y / ih * dh;
                dl->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(255, 0, 0, 220));
                dl->AddCircle(ImVec2(sx, sy), 6, IM_COL32(255, 255, 255, 255), 0, 2);
            }

            // Draw pending init box (cyan)
            if (app.has_init_box) {
                float sx0 = app.canvas_x + app.init_box.x0 / iw * dw;
                float sy0 = app.canvas_y + app.init_box.y0 / ih * dh;
                float sx1 = app.canvas_x + app.init_box.x1 / iw * dw;
                float sy1 = app.canvas_y + app.init_box.y1 / ih * dh;
                dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                            IM_COL32(0, 255, 255, 220), 0, 0, 3);
            }

            // Draw drag-in-progress box (yellow)
            if (app.dragging) {
                float dix, diy;
                if (screen_to_image(app, io.MousePos.x, io.MousePos.y, dix, diy)) {
                    float sx0 = app.canvas_x + app.drag_x0 / iw * dw;
                    float sy0 = app.canvas_y + app.drag_y0 / ih * dh;
                    float sx1 = app.canvas_x + dix / iw * dw;
                    float sy1 = app.canvas_y + diy / ih * dh;
                    dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                                IM_COL32(255, 255, 0, 180), 0, 0, 2);
                }
            }
        } else {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + canvas_max_h * 0.4f);
            ImGui::SetCursorPosX(canvas_max_w * 0.3f);
            ImGui::Text("No video loaded. Use --video <path>");
        }

        // ── Timeline ─────────────────────────────────────────────────────────

        if (app.video_info.n_frames > 0) {
            int n_frames = app.video_info.n_frames;

            ImGui::Spacing();
            ImVec2 tl_pos = ImGui::GetCursorScreenPos();
            float tl_w = canvas_max_w - 16.0f; // small margin
            float tl_bar_h = 14.0f;  // main scrubber bar height

            // Collect unique instance IDs across all timeline entries
            std::vector<int> instance_ids;
            for (const auto& entry : app.timeline) {
                for (const auto& inst : entry.instances) {
                    bool found = false;
                    for (int id : instance_ids) if (id == inst.first) { found = true; break; }
                    if (!found) instance_ids.push_back(inst.first);
                }
            }
            std::sort(instance_ids.begin(), instance_ids.end());

            float band_h = 6.0f;
            float total_h = tl_bar_h + 4.0f + (float)instance_ids.size() * (band_h + 2.0f);
            if (total_h < timeline_h - 16.0f) total_h = timeline_h - 16.0f;

            // Invisible button for click-to-seek
            ImGui::SetCursorScreenPos(tl_pos);
            ImGui::InvisibleButton("timeline", ImVec2(tl_w, total_h));
            bool tl_hovered = ImGui::IsItemHovered();
            bool tl_active  = ImGui::IsItemActive();

            ImDrawList* tl_dl = ImGui::GetWindowDrawList();

            // Background
            tl_dl->AddRectFilled(tl_pos, ImVec2(tl_pos.x + tl_w, tl_pos.y + tl_bar_h),
                                 IM_COL32(40, 40, 40, 255), 3.0f);

            // Processed range highlight
            if (app.timeline_max_frame >= 0) {
                float x1 = tl_pos.x + ((float)(app.timeline_max_frame + 1) / n_frames) * tl_w;
                tl_dl->AddRectFilled(tl_pos, ImVec2(x1, tl_pos.y + tl_bar_h),
                                     IM_COL32(60, 60, 70, 255), 3.0f);
            }

            // Current frame playhead
            float ph_x = tl_pos.x + ((float)app.frame_index / std::max(n_frames - 1, 1)) * tl_w;
            tl_dl->AddRectFilled(ImVec2(ph_x - 1, tl_pos.y),
                                 ImVec2(ph_x + 2, tl_pos.y + tl_bar_h),
                                 IM_COL32(255, 255, 255, 230));

            // Frame number label at playhead
            char frame_lbl[32];
            snprintf(frame_lbl, sizeof(frame_lbl), "%d", app.frame_index);
            tl_dl->AddText(ImVec2(ph_x + 5, tl_pos.y - 1), IM_COL32(200, 200, 200, 220), frame_lbl);

            // Instance presence bands
            float band_y = tl_pos.y + tl_bar_h + 4.0f;
            for (size_t ii = 0; ii < instance_ids.size(); ++ii) {
                int inst_id = instance_ids[ii];
                int ci = inst_id > 0 ? (inst_id - 1) % N_COLORS : (int)(ii % N_COLORS);
                const float* c = INSTANCE_COLORS[ci];
                ImU32 col = IM_COL32((int)(c[0]*255), (int)(c[1]*255), (int)(c[2]*255), 180);
                ImU32 col_dim = IM_COL32((int)(c[0]*80), (int)(c[1]*80), (int)(c[2]*80), 100);

                // Draw dim background for the full band
                tl_dl->AddRectFilled(ImVec2(tl_pos.x, band_y),
                                     ImVec2(tl_pos.x + tl_w, band_y + band_h),
                                     col_dim, 2.0f);

                // Draw bright segments where instance is present
                // Batch consecutive frames into segments for efficiency
                int seg_start = -1;
                for (int f = 0; f < (int)app.timeline.size(); ++f) {
                    bool present = false;
                    for (const auto& p : app.timeline[f].instances)
                        if (p.first == inst_id) { present = true; break; }
                    if (present && seg_start < 0) seg_start = f;
                    if ((!present || f == (int)app.timeline.size() - 1) && seg_start >= 0) {
                        int seg_end = present ? f + 1 : f;
                        float sx0 = tl_pos.x + ((float)seg_start / n_frames) * tl_w;
                        float sx1 = tl_pos.x + ((float)seg_end / n_frames) * tl_w;
                        if (sx1 - sx0 < 2.0f) sx1 = sx0 + 2.0f;
                        tl_dl->AddRectFilled(ImVec2(sx0, band_y), ImVec2(sx1, band_y + band_h),
                                             col, 2.0f);
                        seg_start = -1;
                    }
                }

                // Instance label
                char id_lbl[16];
                snprintf(id_lbl, sizeof(id_lbl), "#%d", inst_id);
                tl_dl->AddText(ImVec2(tl_pos.x + tl_w + 4, band_y - 1), col, id_lbl);

                band_y += band_h + 2.0f;
            }

            // Click-to-seek / drag-to-scrub
            if ((tl_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) || tl_active) {
                float rel = (io.MousePos.x - tl_pos.x) / tl_w;
                rel = std::max(0.0f, std::min(1.0f, rel));
                int target = (int)(rel * (n_frames - 1) + 0.5f);
                if (target != app.frame_index && target >= 0 && target < n_frames) {
                    app.playing = false;
                    if (app.tracker_created) {
                        decode_and_track(app, target);
                    } else {
                        app.frame = sam3_decode_video_frame(app.video_path, target);
                        app.frame_index = target;
                        app.frame_encoded = false;
                        app.result = {};
                    }
                }
            }

            // Tooltip: show frame number on hover
            if (tl_hovered) {
                float rel = (io.MousePos.x - tl_pos.x) / tl_w;
                rel = std::max(0.0f, std::min(1.0f, rel));
                int hover_f = (int)(rel * (n_frames - 1) + 0.5f);
                ImGui::SetTooltip("Frame %d / %d", hover_f, n_frames);
            }
        }

        // ── Bottom panel ─────────────────────────────────────────────────────

        ImGui::SetCursorPosY((float)win_h - panel_h);
        ImGui::Separator();

        ImGui::Checkbox("Show masks", &app.show_masks);
        ImGui::SameLine();
        ImGui::Text("Speed:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("##speed", &app.play_speed, 0.1f, 4.0f, "%.1f");
        ImGui::SameLine();
        if (ImGui::Button("Export frame masks")) {
            export_frame_masks(app);
        }

        // Instance list
        ImGui::Text("Tracked instances:");
        if (app.result.detections.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(none)");
        } else {
            for (size_t d = 0; d < app.result.detections.size(); ++d) {
                const auto& det = app.result.detections[d];
                int ci = det.instance_id > 0 ? (det.instance_id - 1) % N_COLORS : (int)(d % N_COLORS);
                const float* c = INSTANCE_COLORS[ci];
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(c[0], c[1], c[2], 1.0f),
                                   "#%d:%.2f", det.instance_id, det.score);
            }
        }

        // Context-sensitive help
        if (app.init_mode == VMODE_TEXT && !app.visual_only)
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Text mode: auto-detect via text prompt | Click on mask to refine");
        else if (app.init_mode == VMODE_BOX)
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Box mode: drag box to add instance | Click on mask to refine");
        else
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Points mode: left-click +point, right-click -point | Add Instance to confirm");

        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", app.status);

        ImGui::End();

        // ── Render ───────────────────────────────────────────────────────────

        ImGui::Render();
        int fb_w, fb_h;
        SDL_GL_GetDrawableSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────

    if (app.tex_frame) glDeleteTextures(1, &app.tex_frame);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
