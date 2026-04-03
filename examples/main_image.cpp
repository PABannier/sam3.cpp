// sam3_image — Interactive image segmentation example
//
// Usage: sam3_image --model <path.ggml> [--image <path>]
//                   [--threads N] [--no-gpu]
//
// Controls:
//   Mode selector: Points / Box (PVS) / Exemplar (PCS)
//   Left-click on canvas: add positive point (or start box drag).
//   Right-click on canvas: add negative point.
//   Drag on canvas: draw bounding box.
//   Type a text prompt and press [Segment] for PCS.
//   [Clear] resets prompts and masks.
//   [Export] saves masks as PNG files.

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
#include <cmath>
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

static std::string basename_of(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static void apply_theme() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding    = 6.0f;
    s.FrameRounding     = 4.0f;
    s.GrabRounding      = 3.0f;
    s.ScrollbarRounding = 4.0f;
    s.TabRounding       = 4.0f;
    s.ChildRounding     = 4.0f;
    s.PopupRounding     = 4.0f;
    s.FramePadding      = ImVec2(6, 4);
    s.ItemSpacing       = ImVec2(8, 5);
    s.WindowPadding     = ImVec2(10, 10);
    s.FrameBorderSize   = 0.0f;
    s.WindowBorderSize  = 0.0f;
    s.ScrollbarSize     = 12.0f;

    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg]            = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    c[ImGuiCol_ChildBg]             = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    c[ImGuiCol_PopupBg]             = ImVec4(0.10f, 0.10f, 0.13f, 0.96f);
    c[ImGuiCol_Border]              = ImVec4(0.20f, 0.20f, 0.25f, 0.50f);
    c[ImGuiCol_FrameBg]             = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);
    c[ImGuiCol_FrameBgHovered]      = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    c[ImGuiCol_FrameBgActive]       = ImVec4(0.26f, 0.26f, 0.34f, 1.00f);
    c[ImGuiCol_TitleBg]             = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
    c[ImGuiCol_TitleBgActive]       = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
    c[ImGuiCol_MenuBarBg]           = ImVec4(0.10f, 0.10f, 0.13f, 1.00f);
    c[ImGuiCol_ScrollbarBg]         = ImVec4(0.06f, 0.06f, 0.08f, 0.60f);
    c[ImGuiCol_ScrollbarGrab]       = ImVec4(0.28f, 0.28f, 0.34f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]= ImVec4(0.36f, 0.36f, 0.42f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.40f, 0.40f, 0.50f, 1.00f);
    c[ImGuiCol_CheckMark]           = ImVec4(0.45f, 0.60f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrab]          = ImVec4(0.38f, 0.52f, 0.90f, 1.00f);
    c[ImGuiCol_SliderGrabActive]    = ImVec4(0.50f, 0.64f, 1.00f, 1.00f);
    c[ImGuiCol_Button]              = ImVec4(0.18f, 0.20f, 0.28f, 1.00f);
    c[ImGuiCol_ButtonHovered]       = ImVec4(0.30f, 0.34f, 0.48f, 1.00f);
    c[ImGuiCol_ButtonActive]        = ImVec4(0.36f, 0.40f, 0.56f, 1.00f);
    c[ImGuiCol_Header]              = ImVec4(0.18f, 0.20f, 0.28f, 1.00f);
    c[ImGuiCol_HeaderHovered]       = ImVec4(0.26f, 0.30f, 0.42f, 1.00f);
    c[ImGuiCol_HeaderActive]        = ImVec4(0.30f, 0.34f, 0.48f, 1.00f);
    c[ImGuiCol_Separator]           = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);
    c[ImGuiCol_SeparatorHovered]    = ImVec4(0.36f, 0.42f, 0.60f, 1.00f);
    c[ImGuiCol_SeparatorActive]     = ImVec4(0.42f, 0.50f, 0.70f, 1.00f);
    c[ImGuiCol_ResizeGrip]          = ImVec4(0.30f, 0.36f, 0.52f, 0.50f);
    c[ImGuiCol_ResizeGripHovered]   = ImVec4(0.40f, 0.48f, 0.68f, 0.70f);
    c[ImGuiCol_ResizeGripActive]    = ImVec4(0.46f, 0.54f, 0.76f, 0.90f);
    c[ImGuiCol_Tab]                 = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);
    c[ImGuiCol_TabHovered]          = ImVec4(0.30f, 0.34f, 0.48f, 1.00f);
    c[ImGuiCol_TabSelected]         = ImVec4(0.22f, 0.26f, 0.38f, 1.00f);
    c[ImGuiCol_TextSelectedBg]      = ImVec4(0.26f, 0.36f, 0.56f, 0.50f);
    c[ImGuiCol_Text]                = ImVec4(0.90f, 0.90f, 0.94f, 1.00f);
    c[ImGuiCol_TextDisabled]        = ImVec4(0.44f, 0.44f, 0.50f, 1.00f);
}

enum interaction_mode { MODE_POINTS, MODE_BOX_PVS, MODE_EXEMPLAR_PCS };

// Deferred action — we set this to trigger segmentation on the *next* frame,
// so the busy overlay has a chance to render first.
enum pending_action { ACTION_NONE, ACTION_PCS, ACTION_PVS, ACTION_PVS_BOX };

struct app_state {
    // Model
    sam3_params         params;
    std::shared_ptr<sam3_model> model;
    sam3_state_ptr      state;

    // Image
    sam3_image          image;
    GLuint              tex_image   = 0;
    GLuint              tex_overlay = 0;
    bool                image_encoded = false;

    // Interaction mode
    interaction_mode    mode = MODE_POINTS;

    // PCS
    char                text_prompt[256] = {};
    float               score_threshold  = 0.5f;
    float               nms_threshold    = 0.1f;
    std::vector<sam3_box> pos_exemplars;
    std::vector<sam3_box> neg_exemplars;

    // PVS
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    bool                multimask = false;
    sam3_box            pvs_box = {0, 0, 0, 0};
    bool                has_pvs_box = false;

    // Results
    sam3_result         result;
    bool                show_masks = true;

    // Box drawing state
    bool                dragging    = false;
    float               drag_x0     = 0;
    float               drag_y0     = 0;

    // Display
    float               canvas_x    = 0;
    float               canvas_y    = 0;
    float               canvas_w    = 0;
    float               canvas_h    = 0;

    // Status / busy overlay
    char                status[256] = "Ready.";
    bool                busy        = false;
    pending_action      pending     = ACTION_NONE;

    // Model type
    bool                visual_only = false;  // true for SAM2 / SAM3 visual-only
};

static GLuint upload_texture(const uint8_t* data, int w, int h, int ch, GLuint existing = 0) {
    GLuint tex = existing;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GLenum fmt = (ch == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
    return tex;
}

static void build_overlay(app_state& app) {
    if (app.image.data.empty()) return;
    int w = app.image.width;
    int h = app.image.height;

    std::vector<uint8_t> overlay(w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        overlay[4*i+0] = app.image.data[3*i+0];
        overlay[4*i+1] = app.image.data[3*i+1];
        overlay[4*i+2] = app.image.data[3*i+2];
        overlay[4*i+3] = 255;
    }

    if (app.show_masks) {
        for (size_t d = 0; d < app.result.detections.size(); ++d) {
            const auto& det = app.result.detections[d];
            if (det.mask.data.empty()) continue;
            const float* c = INSTANCE_COLORS[d % N_COLORS];
            float alpha = 0.4f;

            int mw = det.mask.width;
            int mh = det.mask.height;
            for (int y = 0; y < std::min(h, mh); ++y) {
                for (int x = 0; x < std::min(w, mw); ++x) {
                    if (det.mask.data[y * mw + x] > 127) {
                        int idx = (y * w + x) * 4;
                        overlay[idx+0] = (uint8_t)(overlay[idx+0] * (1-alpha) + c[0]*255*alpha);
                        overlay[idx+1] = (uint8_t)(overlay[idx+1] * (1-alpha) + c[1]*255*alpha);
                        overlay[idx+2] = (uint8_t)(overlay[idx+2] * (1-alpha) + c[2]*255*alpha);
                    }
                }
            }
        }
    }

    app.tex_overlay = upload_texture(overlay.data(), w, h, 4, app.tex_overlay);
}

static void load_image(app_state& app, const char* path) {
    snprintf(app.status, sizeof(app.status), "Loading image: %s", path);
    app.image = sam3_load_image(path);
    if (app.image.data.empty()) {
        snprintf(app.status, sizeof(app.status), "Failed to load image.");
        return;
    }
    app.tex_image = upload_texture(app.image.data.data(),
                                    app.image.width, app.image.height, 3, app.tex_image);
    app.image_encoded = false;
    app.result = {};
    app.pos_points.clear();
    app.neg_points.clear();
    app.pos_exemplars.clear();
    app.neg_exemplars.clear();
    app.pvs_box = {0, 0, 0, 0};
    app.has_pvs_box = false;

    snprintf(app.status, sizeof(app.status), "Encoding image (%dx%d)...",
             app.image.width, app.image.height);
    app.busy = true;
    if (sam3_encode_image(*app.state, *app.model, app.image)) {
        app.image_encoded = true;
        snprintf(app.status, sizeof(app.status), "Image encoded. Ready for segmentation.");
    } else {
        snprintf(app.status, sizeof(app.status), "Image encoding failed!");
    }
    app.busy = false;
    build_overlay(app);
}

static void run_pcs(app_state& app) {
    if (!app.image_encoded) return;
    if (app.visual_only) {
        snprintf(app.status, sizeof(app.status), "PCS not available for this model.");
        return;
    }
    snprintf(app.status, sizeof(app.status), "Running PCS...");
    app.busy = true;

    sam3_pcs_params pcs;
    pcs.text_prompt      = app.text_prompt;
    pcs.pos_exemplars    = app.pos_exemplars;
    pcs.neg_exemplars    = app.neg_exemplars;
    pcs.score_threshold  = app.score_threshold;
    pcs.nms_threshold    = app.nms_threshold;

    app.result = sam3_segment_pcs(*app.state, *app.model, pcs);
    snprintf(app.status, sizeof(app.status), "PCS: %d detections.",
             (int)app.result.detections.size());
    app.busy = false;
    build_overlay(app);
}

static void run_pvs(app_state& app) {
    if (!app.image_encoded) return;
    if (app.pos_points.empty() && app.neg_points.empty()) return;
    snprintf(app.status, sizeof(app.status), "Running PVS...");
    app.busy = true;

    sam3_pvs_params pvs;
    pvs.pos_points = app.pos_points;
    pvs.neg_points = app.neg_points;
    pvs.multimask  = app.multimask;

    app.result = sam3_segment_pvs(*app.state, *app.model, pvs);
    snprintf(app.status, sizeof(app.status), "PVS: %d masks.",
             (int)app.result.detections.size());
    app.busy = false;
    build_overlay(app);
}

static void run_pvs_box(app_state& app) {
    if (!app.image_encoded) return;
    if (!app.has_pvs_box) return;
    snprintf(app.status, sizeof(app.status), "Running PVS (box)...");
    app.busy = true;

    sam3_pvs_params pvs;
    pvs.pos_points = app.pos_points;
    pvs.neg_points = app.neg_points;
    pvs.box        = app.pvs_box;
    pvs.use_box    = true;
    pvs.multimask  = app.multimask;

    app.result = sam3_segment_pvs(*app.state, *app.model, pvs);
    snprintf(app.status, sizeof(app.status), "PVS (box): %d masks.",
             (int)app.result.detections.size());
    app.busy = false;
    build_overlay(app);
}

static void export_masks(const app_state& app) {
    for (size_t i = 0; i < app.result.detections.size(); ++i) {
        char path[256];
        snprintf(path, sizeof(path), "mask_%02d.png", (int)i);
        if (sam3_save_mask(app.result.detections[i].mask, path)) {
            fprintf(stderr, "Exported %s\n", path);
        }
    }
}

// Convert screen position to image coordinates
static bool screen_to_image(const app_state& app, float sx, float sy,
                             float& ix, float& iy) {
    if (app.canvas_w <= 0 || app.canvas_h <= 0) return false;
    ix = (sx - app.canvas_x) / app.canvas_w * app.image.width;
    iy = (sy - app.canvas_y) / app.canvas_h * app.image.height;
    return ix >= 0 && iy >= 0 && ix < app.image.width && iy < app.image.height;
}

// Draw a spinning indicator + message as a centered overlay
static void draw_busy_overlay(const char* message) {
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 center(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);

    // Dim background
    ImGui::GetForegroundDrawList()->AddRectFilled(
        ImVec2(0, 0), io.DisplaySize, IM_COL32(0, 0, 0, 140));

    // Spinner
    float t = (float)ImGui::GetTime();
    float radius = 20.0f;
    int num_segments = 12;
    for (int i = 0; i < num_segments; ++i) {
        float angle = t * 5.0f + i * (2.0f * 3.14159f / num_segments);
        float alpha = (float)(i + 1) / num_segments;
        float x = center.x + cosf(angle) * radius;
        float y = center.y - 30.0f + sinf(angle) * radius;
        ImGui::GetForegroundDrawList()->AddCircleFilled(
            ImVec2(x, y), 3.0f,
            IM_COL32(255, 255, 255, (int)(alpha * 255)));
    }

    // Text below spinner
    ImVec2 text_size = ImGui::CalcTextSize(message);
    ImGui::GetForegroundDrawList()->AddText(
        ImVec2(center.x - text_size.x * 0.5f, center.y + 10.0f),
        IM_COL32(255, 255, 255, 255), message);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    app_state app;
    app.params.n_threads = 4;
    app.params.use_gpu   = true;

    const char* image_path = nullptr;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            app.params.model_path = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i+1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) {
            app.params.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
            app.params.use_gpu = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model <path.ggml> [--image <path>]\n"
                "          [--threads N] [--no-gpu]\n", argv[0]);
            return 0;
        }
    }

    if (app.params.model_path.empty()) {
        fprintf(stderr, "Error: --model is required.\n");
        return 1;
    }

    fprintf(stderr, "Using %d threads\n", app.params.n_threads);

    // ── Init SDL2 + OpenGL ───────────────────────────────────────────────────

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
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

    std::string init_title = basename_of(app.params.model_path);
    if (image_path) {
        init_title = basename_of(image_path) + " \xe2\x80\x94 " + init_title;
    }

    SDL_Window* window = SDL_CreateWindow(
        init_title.c_str(),
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

    // ── HiDPI scaling ────────────────────────────────────────────────────────

    int fb_w_init, fb_h_init, win_w_init, win_h_init;
    SDL_GL_GetDrawableSize(window, &fb_w_init, &fb_h_init);
    SDL_GetWindowSize(window, &win_w_init, &win_h_init);
    float dpi_scale = (win_w_init > 0) ? (float)fb_w_init / (float)win_w_init : 1.0f;

    // ── Init ImGui ───────────────────────────────────────────────────────────

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    apply_theme();

    float font_size = 15.0f * dpi_scale;
    ImFontConfig font_cfg;
    font_cfg.OversampleH = 2;
    font_cfg.OversampleV = 2;
    ImFont* font = nullptr;
#ifdef __APPLE__
    font = io.Fonts->AddFontFromFileTTF("/System/Library/Fonts/SFNS.ttf", font_size, &font_cfg);
#elif defined(_WIN32)
    font = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", font_size, &font_cfg);
#else
    font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size, &font_cfg);
#endif
    if (!font) {
        font_cfg.SizePixels = 13.0f * dpi_scale;
        io.Fonts->AddFontDefault(&font_cfg);
    }
    io.FontGlobalScale = 1.0f / dpi_scale;

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
        fprintf(stderr, "Model: %s — PCS (text) mode disabled\n",
                mt == SAM3_MODEL_SAM2 ? "SAM2" : "SAM3 visual-only");
        app.mode = MODE_POINTS;
        snprintf(app.status, sizeof(app.status),
                 "%s model loaded (PVS only). Open an image or use --image.",
                 mt == SAM3_MODEL_SAM2 ? "SAM2" : "SAM3 visual-only");
    } else {
        snprintf(app.status, sizeof(app.status), "Model loaded. Open an image or use --image.");
    }

    // Load initial image if provided
    if (image_path) {
        load_image(app, image_path);
    }

    // ── Main loop ────────────────────────────────────────────────────────────

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) running = false;

            // Handle drag-and-drop
            if (event.type == SDL_DROPFILE) {
                std::string dropped = event.drop.file;
                load_image(app, event.drop.file);
                SDL_free(event.drop.file);
                std::string title = basename_of(dropped) + " \xe2\x80\x94 " + basename_of(app.params.model_path);
                SDL_SetWindowTitle(window, title.c_str());
            }
        }

        // ── Execute deferred segmentation (runs after the busy overlay frame) ─

        if (app.pending != ACTION_NONE) {
            pending_action act = app.pending;
            app.pending = ACTION_NONE;
            switch (act) {
                case ACTION_PCS:     run_pcs(app); break;
                case ACTION_PVS:     run_pvs(app); break;
                case ACTION_PVS_BOX: run_pvs_box(app); break;
                default: break;
            }
        }

        // ── ImGui frame ──────────────────────────────────────────────────────

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // Full-window panel
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);
        ImGui::SetNextWindowSize(ImVec2((float)win_w, (float)win_h));
        ImGui::Begin("sam3", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        // ── Top bar: text prompt + buttons ───────────────────────────────────

        if (!app.visual_only) {
            ImGui::Text("Text prompt:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(300);
            bool enter_pressed = ImGui::InputText("##prompt", app.text_prompt,
                                                   sizeof(app.text_prompt),
                                                   ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::SameLine();
            if ((ImGui::Button("Segment") || enter_pressed) && app.image_encoded) {
                app.busy = true;
                app.pending = ACTION_PCS;
                snprintf(app.status, sizeof(app.status), "Segmenting...");
            }
            ImGui::SameLine();
        }
        if (ImGui::Button("Clear")) {
            app.result = {};
            app.pos_points.clear();
            app.neg_points.clear();
            app.pos_exemplars.clear();
            app.neg_exemplars.clear();
            app.pvs_box = {0, 0, 0, 0};
            app.has_pvs_box = false;
            app.text_prompt[0] = '\0';
            build_overlay(app);
        }
        ImGui::SameLine();
        if (ImGui::Button("Export masks")) {
            export_masks(app);
        }

        // Mode selector
        ImGui::Text("Mode:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Points", app.mode == MODE_POINTS)) app.mode = MODE_POINTS;
        ImGui::SameLine();
        if (ImGui::RadioButton("Box (PVS)", app.mode == MODE_BOX_PVS)) app.mode = MODE_BOX_PVS;
        ImGui::SameLine();
        if (!app.visual_only) {
            if (ImGui::RadioButton("Exemplar (PCS)", app.mode == MODE_EXEMPLAR_PCS))
                app.mode = MODE_EXEMPLAR_PCS;
        } else {
            ImGui::BeginDisabled();
            ImGui::RadioButton("Exemplar (PCS)", false);
            ImGui::EndDisabled();
        }

        // ── Image canvas ─────────────────────────────────────────────────────

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float panel_h = 145.0f;
        float canvas_max_h = avail.y - panel_h;
        float canvas_max_w = avail.x;

        if (!app.image.data.empty() && app.tex_overlay) {
            float iw = (float)app.image.width;
            float ih = (float)app.image.height;
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

            // Draw the image via an InvisibleButton so we get proper hover/click
            ImGui::SetCursorScreenPos(pos);
            ImGui::InvisibleButton("canvas", ImVec2(dw, dh));
            bool canvas_hovered = ImGui::IsItemHovered();
            bool canvas_active  = ImGui::IsItemActive();

            // Draw the texture manually on the draw list
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddImage((ImTextureID)(intptr_t)app.tex_overlay,
                         pos, ImVec2(pos.x + dw, pos.y + dh));

            // ── Canvas mouse interaction ─────────────────────────────────────

            if (canvas_hovered && !app.busy && app.image_encoded) {
                float mx = io.MousePos.x;
                float my = io.MousePos.y;
                float ix, iy;

                // Left button down → start drag
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        app.dragging = true;
                        app.drag_x0  = ix;
                        app.drag_y0  = iy;
                    }
                }

                // Right click → negative point
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    if (screen_to_image(app, mx, my, ix, iy)) {
                        app.neg_points.push_back({ix, iy});
                        app.busy = true;
                        app.pending = app.has_pvs_box ? ACTION_PVS_BOX : ACTION_PVS;
                        snprintf(app.status, sizeof(app.status), "Segmenting...");
                    }
                }
            }

            // Left button release → finish drag or click
            if (app.dragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                float mx = io.MousePos.x;
                float my = io.MousePos.y;
                float ix, iy;
                if (screen_to_image(app, mx, my, ix, iy)) {
                    float dx = ix - app.drag_x0;
                    float dy = iy - app.drag_y0;
                    if (dx*dx + dy*dy > 25.0f) {
                        // Drag → box
                        sam3_box box;
                        box.x0 = std::min(app.drag_x0, ix);
                        box.y0 = std::min(app.drag_y0, iy);
                        box.x1 = std::max(app.drag_x0, ix);
                        box.y1 = std::max(app.drag_y0, iy);
                        if (app.mode == MODE_EXEMPLAR_PCS) {
                            app.pos_exemplars.push_back(box);
                        } else {
                            // Points or Box mode → PVS box
                            app.pvs_box = box;
                            app.has_pvs_box = true;
                            app.busy = true;
                            app.pending = ACTION_PVS_BOX;
                            snprintf(app.status, sizeof(app.status), "Segmenting...");
                        }
                    } else {
                        // Click → positive point
                        app.pos_points.push_back({app.drag_x0, app.drag_y0});
                        app.busy = true;
                        app.pending = app.has_pvs_box ? ACTION_PVS_BOX : ACTION_PVS;
                        snprintf(app.status, sizeof(app.status), "Segmenting...");
                    }
                }
                app.dragging = false;
            }

            // ── Draw annotations on canvas ───────────────────────────────────

            // Points
            for (const auto& p : app.pos_points) {
                float sx = app.canvas_x + p.x / iw * dw;
                float sy = app.canvas_y + p.y / ih * dh;
                dl->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(0, 255, 0, 220));
                dl->AddCircle(ImVec2(sx, sy), 6, IM_COL32(255, 255, 255, 255), 0, 2);
            }
            for (const auto& p : app.neg_points) {
                float sx = app.canvas_x + p.x / iw * dw;
                float sy = app.canvas_y + p.y / ih * dh;
                dl->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(255, 0, 0, 220));
                dl->AddCircle(ImVec2(sx, sy), 6, IM_COL32(255, 255, 255, 255), 0, 2);
            }

            // Exemplar boxes (green)
            for (const auto& b : app.pos_exemplars) {
                float sx0 = app.canvas_x + b.x0 / iw * dw;
                float sy0 = app.canvas_y + b.y0 / ih * dh;
                float sx1 = app.canvas_x + b.x1 / iw * dw;
                float sy1 = app.canvas_y + b.y1 / ih * dh;
                dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                            IM_COL32(0, 255, 0, 200), 0, 0, 2);
            }

            // PVS bounding box (cyan)
            if (app.has_pvs_box) {
                float sx0 = app.canvas_x + app.pvs_box.x0 / iw * dw;
                float sy0 = app.canvas_y + app.pvs_box.y0 / ih * dh;
                float sx1 = app.canvas_x + app.pvs_box.x1 / iw * dw;
                float sy1 = app.canvas_y + app.pvs_box.y1 / ih * dh;
                dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                            IM_COL32(0, 255, 255, 220), 0, 0, 3);
            }

            // Drag-in-progress box (yellow)
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

            // Detection boxes + labels
            for (size_t d = 0; d < app.result.detections.size(); ++d) {
                const auto& det = app.result.detections[d];
                const float* c = INSTANCE_COLORS[d % N_COLORS];
                ImU32 col = IM_COL32((int)(c[0]*255), (int)(c[1]*255),
                                     (int)(c[2]*255), 200);
                float sx0 = app.canvas_x + det.box.x0 / iw * dw;
                float sy0 = app.canvas_y + det.box.y0 / ih * dh;
                float sx1 = app.canvas_x + det.box.x1 / iw * dw;
                float sy1 = app.canvas_y + det.box.y1 / ih * dh;
                dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1), col, 0, 0, 2);

                char label[64];
                snprintf(label, sizeof(label), "#%d %.2f", det.instance_id, det.score);
                dl->AddText(ImVec2(sx0 + 2, sy0 + 2), col, label);
            }
        } else {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + canvas_max_h * 0.4f);
            ImGui::SetCursorPosX(canvas_max_w * 0.3f);
            ImGui::TextWrapped("Drag and drop an image here, or use --image <path>");
        }

        // ── Bottom panel ─────────────────────────────────────────────────────

        ImGui::SetCursorPosY((float)win_h - panel_h);
        ImGui::Separator();

        ImGui::Text("Detections: %d instances", (int)app.result.detections.size());

        ImGui::Text("Score threshold:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("##score", &app.score_threshold, 0.0f, 1.0f, "%.2f");

        ImGui::Checkbox("Show masks", &app.show_masks);
        ImGui::SameLine();
        ImGui::Checkbox("Multi-mask (PVS)", &app.multimask);

        // Context-sensitive help
        if (app.mode == MODE_POINTS)
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Left-click: +point | Right-click: -point | Drag: bounding box");
        else if (app.mode == MODE_BOX_PVS)
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Drag: bounding box (PVS) | Left-click: +point | Right-click: -point");
        else if (!app.visual_only)
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Drag: exemplar box | Type text and press Segment");

        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", app.status);

        // Detection list
        if (!app.result.detections.empty()) {
            ImGui::SameLine();
            ImGui::Text("  |");
            for (size_t d = 0; d < std::min(app.result.detections.size(), (size_t)8); ++d) {
                ImGui::SameLine();
                const auto& det = app.result.detections[d];
                const float* c = INSTANCE_COLORS[d % N_COLORS];
                ImGui::TextColored(ImVec4(c[0], c[1], c[2], 1.0f),
                                   "#%d:%.2f", det.instance_id, det.score);
            }
        }

        ImGui::End();

        // ── Busy overlay (drawn on foreground, over everything) ──────────────

        if (app.busy) {
            draw_busy_overlay("Segmenting...");
        }

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

    if (app.tex_image)   glDeleteTextures(1, &app.tex_image);
    if (app.tex_overlay) glDeleteTextures(1, &app.tex_overlay);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
