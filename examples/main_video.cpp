// sam3_video — Interactive video segmentation + tracking example
//
// Usage: sam3_video --model <path.ggml> --video <path> [--tokenizer <dir>]
//
// Controls:
//   Type a text prompt → auto-detect instances on each frame.
//   [Play]/[Pause]/[Step] for playback control.
//   Left-click: add positive refinement point for nearest instance.
//   Right-click: add negative refinement point.
//   [Export] saves mask PNGs per frame.

#include "sam3.h"

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
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

    // Tracking
    char                    text_prompt[256] = {};
    sam3_video_params       track_params;
    sam3_result             result;
    bool                    tracker_created = false;

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
    app.track_params.text_prompt = app.text_prompt;
    app.tracker = sam3_create_tracker(*app.model, app.track_params);
    app.tracker_created = (app.tracker != nullptr);
    if (app.tracker_created) {
        snprintf(app.status, sizeof(app.status), "Tracker created. Press Play.");
    } else {
        snprintf(app.status, sizeof(app.status), "Failed to create tracker.");
    }
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
        app.result = sam3_track_frame(*app.tracker, *app.state, *app.model, app.frame);
        snprintf(app.status, sizeof(app.status), "Frame %d/%d — %d objects tracked",
                 fi, app.video_info.n_frames, (int)app.result.detections.size());
    } else {
        // Just encode and show the frame without tracking
        sam3_encode_image(*app.state, *app.model, app.frame);
        app.result = {};
        snprintf(app.status, sizeof(app.status), "Frame %d/%d — no tracker active",
                 fi, app.video_info.n_frames);
    }
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
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i+1 < argc) {
            app.params.tokenizer_dir = argv[++i];
        } else if (strcmp(argv[i], "--video") == 0 && i+1 < argc) {
            app.video_path = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) {
            app.params.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
            app.params.use_gpu = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model <path.ggml> --video <path> [--tokenizer <dir>]\n"
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

    // ── Load video info ──────────────────────────────────────────────────────

    if (!app.video_path.empty()) {
        app.video_info = sam3_get_video_info(app.video_path);
        if (app.video_info.n_frames > 0) {
            snprintf(app.status, sizeof(app.status),
                     "Video: %dx%d, %d frames, %.1f fps. Enter a text prompt and press Start.",
                     app.video_info.width, app.video_info.height,
                     app.video_info.n_frames, app.video_info.fps);
            // Decode first frame for preview
            app.frame = sam3_decode_video_frame(app.video_path, 0);
            app.frame_index = 0;
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

            // Mouse on canvas for refinement
            if (!io.WantCaptureMouse && !app.frame.data.empty() && app.tracker_created) {
                float mx = io.MousePos.x, my = io.MousePos.y;
                float ix, iy;

                if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        // Find nearest instance and refine
                        int best_id = -1;
                        for (const auto& det : app.result.detections) {
                            if (det.mask.data.empty()) continue;
                            int px = (int)ix, py = (int)iy;
                            if (px >= 0 && px < det.mask.width &&
                                py >= 0 && py < det.mask.height &&
                                det.mask.data[py * det.mask.width + px] > 127) {
                                best_id = det.instance_id;
                                break;
                            }
                        }
                        if (best_id >= 0) {
                            std::vector<sam3_point> pos = {{ix, iy}};
                            std::vector<sam3_point> neg;
                            sam3_refine_instance(*app.tracker, *app.state, *app.model,
                                                 best_id, pos, neg);
                            snprintf(app.status, sizeof(app.status),
                                     "Refined instance #%d with positive point", best_id);
                        }
                    }
                }
                if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        // Find nearest instance and add negative point
                        int best_id = -1;
                        for (const auto& det : app.result.detections) {
                            if (det.mask.data.empty()) continue;
                            int px = (int)ix, py = (int)iy;
                            if (px >= 0 && px < det.mask.width &&
                                py >= 0 && py < det.mask.height &&
                                det.mask.data[py * det.mask.width + px] > 127) {
                                best_id = det.instance_id;
                                break;
                            }
                        }
                        if (best_id >= 0) {
                            std::vector<sam3_point> pos;
                            std::vector<sam3_point> neg = {{ix, iy}};
                            sam3_refine_instance(*app.tracker, *app.state, *app.model,
                                                 best_id, pos, neg);
                            snprintf(app.status, sizeof(app.status),
                                     "Refined instance #%d with negative point", best_id);
                        }
                    }
                }
            }

            // Keyboard shortcuts
            if (event.type == SDL_KEYDOWN && !io.WantCaptureKeyboard) {
                if (event.key.keysym.sym == SDLK_SPACE) {
                    app.playing = !app.playing;
                } else if (event.key.keysym.sym == SDLK_RIGHT) {
                    // Step forward
                    if (app.frame_index + 1 < app.video_info.n_frames) {
                        decode_and_track(app, app.frame_index + 1);
                    }
                } else if (event.key.keysym.sym == SDLK_LEFT) {
                    // Step backward (re-decode only, tracking is forward-only)
                    if (app.frame_index > 0) {
                        app.frame = sam3_decode_video_frame(app.video_path, app.frame_index - 1);
                        app.frame_index--;
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

        ImGui::Text("Text:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(250);
        ImGui::InputText("##prompt", app.text_prompt, sizeof(app.text_prompt));
        ImGui::SameLine();

        if (!app.tracker_created) {
            if (ImGui::Button("Start tracking")) {
                create_tracker(app);
                if (app.tracker_created && !app.frame.data.empty()) {
                    decode_and_track(app, 0);
                }
            }
        } else {
            if (app.playing) {
                if (ImGui::Button("Pause")) app.playing = false;
            } else {
                if (ImGui::Button("Play")) {
                    app.playing = true;
                    app.last_frame_time = SDL_GetTicks();
                }
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
            app.tracker.reset();
            app.result = {};
            app.frame_index = 0;
            if (!app.video_path.empty()) {
                app.frame = sam3_decode_video_frame(app.video_path, 0);
            }
            snprintf(app.status, sizeof(app.status), "Tracker reset.");
        }

        ImGui::Text("Frame: %d/%d   FPS: %.1f   Objects: %d   Speed: %.1fx",
                     app.frame_index, app.video_info.n_frames,
                     app.video_info.fps, (int)app.result.detections.size(),
                     app.play_speed);

        // ── Video canvas ─────────────────────────────────────────────────────

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float panel_h = 120.0f;
        float canvas_max_h = avail.y - panel_h;
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
            ImGui::Image((ImTextureID)(intptr_t)app.tex_frame, ImVec2(dw, dh));

            // Draw instance labels on canvas
            ImDrawList* dl = ImGui::GetWindowDrawList();
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
        } else {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + canvas_max_h * 0.4f);
            ImGui::SetCursorPosX(canvas_max_w * 0.3f);
            ImGui::Text("No video loaded. Use --video <path>");
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
