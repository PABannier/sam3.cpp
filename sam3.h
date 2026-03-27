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

// Add a new instance to the tracker from PVS prompts (points/box) on the
// current frame.  The image must already be encoded (via sam3_track_frame or
// sam3_encode_image).  Returns assigned instance_id, or -1 on failure.
int sam3_tracker_add_instance(sam3_tracker         & tracker,
                              sam3_state            & state,
                              const sam3_model      & model,
                              const sam3_pvs_params & pvs_params);

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

// Test-only: run the full phase 5 detector path (fusion encoder + DETR decoder
// + dot-product scoring + segmentation head) on an already encoded image state
// and dump standard intermediate tensors to <output_dir>/<tensor_name>.{bin,shape}.
bool sam3_test_dump_phase5(const sam3_model & model,
                           const sam3_state & state,
                           const std::vector<int32_t> & token_ids,
                           const std::string & output_dir,
                           int n_threads = 4);

// Test-only: run the phase 5 detector path from pre-dumped phase inputs
// instead of re-running the image/text encoders. This isolates the detector
// numerics from earlier phases and is intended for cross-phase regression tests.
bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model & model,
                                           const std::vector<int32_t> & token_ids,
                                           const std::string & prephase_ref_dir,
                                           const std::string & phase5_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads = 4);

// Test-only: run the phase 6 prompt encoder + SAM decoder on an already
// encoded tracker image state and dump standard intermediate tensors to
// <output_dir>/<tensor_name>.{bin,shape}.
bool sam3_test_dump_phase6(const sam3_model & model,
                           const sam3_state & state,
                           const sam3_pvs_params & params,
                           const std::string & output_dir,
                           int n_threads = 4);

// Test-only: run the phase 6 prompt encoder + SAM decoder from pre-dumped
// phase 3 tracker features. This isolates phase 6 numerics from earlier phases
// and is intended to be reused by later tracker-stage tests.
bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model & model,
                                           const std::string & prephase_ref_dir,
                                           const sam3_pvs_params & params,
                                           const std::string & output_dir,
                                           int n_threads = 4);

// Test-only: run the phase 7 tracker subgraph from pre-dumped case inputs and
// dump standard intermediate tensors to <output_dir>/<tensor_name>.{bin,shape}.
// The case directory is produced by tests/dump_phase7_reference.py.
bool sam3_test_dump_phase7_from_ref_inputs(const sam3_model & model,
                                           const std::string & case_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads = 4);

// Test-only: run the geometry encoder from pre-computed backbone features and
// dump intermediate tensors to <output_dir>/<tensor_name>.{bin,shape}.
// Tests exemplar box coordinate encoding against Python reference.
bool sam3_test_dump_geom_enc(const sam3_model   & model,
                              const std::string  & prephase_ref_dir,
                              const sam3_pcs_params & params,
                              const std::string  & output_dir,
                              int                  n_threads = 4);

// Test-only: run ONLY the fusion encoder (6 layers) from pre-dumped inputs
// (image features, pos encoding, prompt tokens, attn bias).  Dumps per-layer
// outputs for isolated fenc debugging.  Input tensors should come from the
// actual Python package via tests/dump_fenc_from_package.py.
bool sam3_test_fenc_only(const sam3_model  & model,
                          const std::string & ref_dir,
                          const std::string & output_dir,
                          int                 n_threads = 4);

// ─── Debug: dump state tensors to files for verification ───

bool sam3_dump_state_tensor(const sam3_state & state,
                             const std::string & tensor_name,
                             const std::string & output_path);

// Test-only: encode an image from pre-preprocessed float data (CHW layout, already
// resized to img_size x img_size and normalized).  This bypasses the C++ preprocessing
// so that numerical comparisons against the Python reference are not polluted by
// differences in image resize implementations.
bool sam3_encode_image_from_preprocessed(sam3_state       & state,
                                          const sam3_model & model,
                                          const float      * chw_data,
                                          int                img_size);
