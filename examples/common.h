// Shared GUI helpers for the sam3_image and sam3_video examples.
#pragma once

#include <SDL.h>
#include <SDL_opengl.h>

#include <imgui.h>

#include <cstdint>
#include <string>

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
