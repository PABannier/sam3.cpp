#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ─── Forward declarations (opaque, defined in sam3.cpp) ───

struct sam3_model;
struct sam3_state;
struct sam3_tracker;

// Custom deleters so unique_ptr works with forward-declared opaque types.
struct sam3_state_deleter   { void operator()(sam3_state * p) const; };
struct sam3_tracker_deleter { void operator()(sam3_tracker * p) const; };

using sam3_state_ptr   = std::unique_ptr<sam3_state,   sam3_state_deleter>;
using sam3_tracker_ptr = std::unique_ptr<sam3_tracker,  sam3_tracker_deleter>;

// ─── Public data types ───

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int width    = 0;
    int height   = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int   width       = 0;
    int   height      = 0;
    float iou_score   = 0.0f;
    float obj_score   = 0.0f;
    int   instance_id = -1;
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box  box;
    float     score     = 0.0f;
    float     iou_score = 0.0f;
    int       instance_id = -1;
    sam3_mask  mask;
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

// ─── Parameters ───

struct sam3_params {
    std::string model_path;
    std::string tokenizer_dir;   // directory containing vocab.json + merges.txt
    int         n_threads = 4;
    bool        use_gpu   = true;
    int         seed      = 42;
};

struct sam3_pcs_params {
    std::string            text_prompt;
    std::vector<sam3_box>  pos_exemplars;
    std::vector<sam3_box>  neg_exemplars;
    float                  score_threshold = 0.5f;
    float                  nms_threshold   = 0.1f;
};

struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    sam3_box                box      = {0, 0, 0, 0};
    bool                    use_box  = false;
    bool                    multimask = false;
};

struct sam3_video_params {
    std::string text_prompt;
    float       score_threshold     = 0.5f;
    float       nms_threshold       = 0.1f;
    float       assoc_iou_threshold = 0.1f;
    int         hotstart_delay      = 15;
    int         max_keep_alive      = 30;
    int         recondition_every   = 16;
    int         fill_hole_area      = 16;
};

struct sam3_video_info {
    int   width    = 0;
    int   height   = 0;
    int   n_frames = 0;
    float fps      = 0.0f;
};

// ─── Model lifecycle ───

std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params);
void sam3_free_model(sam3_model & model);

// ─── Inference state ───

sam3_state_ptr sam3_create_state(const sam3_model & model,
                                const sam3_params & params);
void sam3_free_state(sam3_state & state);

// ─── Image backbone (call once per image) ───

bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image);

// ─── Image segmentation ───

sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params);

sam3_result sam3_segment_pvs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pvs_params  & params);

// ─── Video tracking ───

sam3_tracker_ptr sam3_create_tracker(const sam3_model       & model,
                                    const sam3_video_params & params);

sam3_result sam3_track_frame(sam3_tracker     & tracker,
                             sam3_state       & state,
                             const sam3_model & model,
                             const sam3_image & frame);

bool sam3_refine_instance(sam3_tracker                   & tracker,
                          sam3_state                     & state,
                          const sam3_model               & model,
                          int                              instance_id,
                          const std::vector<sam3_point>  & pos_points,
                          const std::vector<sam3_point>  & neg_points);

int  sam3_tracker_frame_index(const sam3_tracker & tracker);
void sam3_tracker_reset(sam3_tracker & tracker);

// ─── Utility ───

sam3_image      sam3_load_image(const std::string & path);
bool            sam3_save_mask(const sam3_mask & mask, const std::string & path);
sam3_image      sam3_decode_video_frame(const std::string & video_path, int frame_index);
sam3_video_info sam3_get_video_info(const std::string & video_path);

// ─── Tokenizer (standalone, does not require model weights) ───

bool                  sam3_test_load_tokenizer(const std::string & dir);
std::vector<int32_t>  sam3_test_tokenize(const std::string & text);

// Test-only: run the text encoder on fixed token IDs and dump standard
// intermediate tensors to <output_dir>/<tensor_name>.{bin,shape}.
bool sam3_test_dump_text_encoder(const sam3_model & model,
                                 const std::vector<int32_t> & token_ids,
                                 const std::string & output_dir,
                                 int n_threads = 4);

// ─── Debug: dump state tensors to files for verification ───

bool sam3_dump_state_tensor(const sam3_state & state,
                             const std::string & tensor_name,
                             const std::string & output_path);
