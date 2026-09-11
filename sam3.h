#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define SAM3_VERSION_MAJOR 1
#define SAM3_VERSION_MINOR 0
#define SAM3_VERSION_PATCH 0
#define SAM3_VERSION       "1.0.0"

/*
** ── Forward Declarations ─────────────────────────────────────────────────
*/

struct sam3_model;
struct sam3_state;
struct sam3_tracker;

/* Custom deleters so unique_ptr works with forward-declared opaque types. */
struct sam3_state_deleter   { void operator()(sam3_state * p) const; };
struct sam3_tracker_deleter { void operator()(sam3_tracker * p) const; };

using sam3_state_ptr   = std::unique_ptr<sam3_state,   sam3_state_deleter>;
using sam3_tracker_ptr = std::unique_ptr<sam3_tracker,  sam3_tracker_deleter>;

/*
** ── Model Type ──────────────────────────────────────────────────────────
*/

enum sam3_model_type {
    SAM3_MODEL_SAM3        = 0,  // Full SAM3 (ViT + detector + tracker)
    SAM3_MODEL_SAM2        = 2,  // SAM2 (Hiera + tracker, no text/detector)
    SAM3_MODEL_EDGETAM     = 3,  // EdgeTAM (RepViT + Perceiver, no text/detector)
};

/*****************************************************************************
** Public Data Types
**
** Geometry primitives, images, masks, and detection results.
*****************************************************************************/

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
    std::vector<float> sam_token;  // raw SAM decoder output token (for obj_ptr)
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

/*****************************************************************************
** Parameters
**
** Configuration for model loading, segmentation, and video tracking.
*****************************************************************************/

struct sam3_params {
    std::string model_path;
    int         n_threads       = 4;
    bool        use_gpu         = true;
    int         encode_img_size = 0;  // 0 = model default; override input resolution
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

/*****************************************************************************
** Public API
**
** Model lifecycle, image encoding, segmentation, and video tracking.
*****************************************************************************/

/*
** ── Model Lifecycle ──────────────────────────────────────────────────────
*/

/*
** Load a SAM3 model from the file specified in params.model_path.
** Returns nullptr on failure.
*/
std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params);

/* Free all resources held by a loaded model. */
void sam3_free_model(sam3_model & model);

/* Returns true if the model was loaded as visual-only (no text/detector path).
** SAM2 models are always considered visual-only. */
bool sam3_is_visual_only(const sam3_model & model);

/* Returns the model type (SAM2 or SAM3). */
sam3_model_type sam3_get_model_type(const sam3_model & model);

/*
** ── Inference State ──────────────────────────────────────────────────────
*/

/* Allocate inference state (backbone caches, PE buffers). */
sam3_state_ptr sam3_create_state(const sam3_model & model,
                                const sam3_params & params);

/* Free inference state and its GPU buffers. */
void sam3_free_state(sam3_state & state);

/*
** ── Image Backbone ───────────────────────────────────────────────────────
*/

/*
** Encode an image through the ViT backbone and FPN neck.
** Call once per image before segmentation or tracking.
** Returns true on success, false on failure.
*/
bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image);

/*
** ── Image Segmentation ──────────────────────────────────────────────────
*/

/* Segment using text prompt + exemplar boxes (PCS path). */
sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params);

/* Segment using point/box prompts (PVS path). */
sam3_result sam3_segment_pvs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pvs_params  & params);

/*
** ── Video Tracking ──────────────────────────────────────────────────────
*/

/* Create a tracker for text-prompted video segmentation. */
sam3_tracker_ptr sam3_create_tracker(const sam3_model       & model,
                                    const sam3_video_params & params);

/* Encode a frame, detect objects, and update tracked instances. */
sam3_result sam3_track_frame(sam3_tracker     & tracker,
                             sam3_state       & state,
                             const sam3_model & model,
                             const sam3_image & frame);

/* Refine a tracked instance with interactive point prompts. */
bool sam3_refine_instance(sam3_tracker                   & tracker,
                          sam3_state                     & state,
                          const sam3_model               & model,
                          int                              instance_id,
                          const std::vector<sam3_point>  & pos_points,
                          const std::vector<sam3_point>  & neg_points);

/*
** Add a new instance to the tracker from PVS prompts (points/box) on the
** current frame.  The image must already be encoded (via sam3_track_frame
** or sam3_encode_image).  Returns assigned instance_id, or -1 on failure.
*/
int sam3_tracker_add_instance(sam3_tracker         & tracker,
                              sam3_state            & state,
                              const sam3_model      & model,
                              const sam3_pvs_params & pvs_params);

/* Return the current frame index of the tracker. */
int  sam3_tracker_frame_index(const sam3_tracker & tracker);

/* Reset the tracker, clearing all instances and memory. */
void sam3_tracker_reset(sam3_tracker & tracker);

/*
** ── Visual-Only Video Tracking ──────────────────────────────────────────
*/

struct sam3_visual_track_params {
    float assoc_iou_threshold = 0.1f;
    int   max_keep_alive      = 30;
    int   recondition_every   = 16;
    int   fill_hole_area      = 16;
};

/*
** Create a tracker for visual-only models.  Instances are added manually
** via sam3_tracker_add_instance().
*/
sam3_tracker_ptr sam3_create_visual_tracker(
    const sam3_model               & model,
    const sam3_visual_track_params & params);

/*
** Propagate all tracked instances to the next frame (no detection step).
** The image is encoded, then each tracked instance is propagated via
** memory attention + SAM mask decode, and the memory bank is updated.
*/
sam3_result sam3_propagate_frame(
    sam3_tracker     & tracker,
    sam3_state       & state,
    const sam3_model & model,
    const sam3_image & frame);

/*
** ── Utility ─────────────────────────────────────────────────────────────
*/

sam3_image      sam3_load_image(const std::string & path);
bool            sam3_save_mask(const sam3_mask & mask, const std::string & path);
sam3_image      sam3_decode_video_frame(const std::string & video_path, int frame_index);
sam3_video_info sam3_get_video_info(const std::string & video_path);

/*
** ── Profiling ───────────────────────────────────────────────────────────
*/

/*
** Profile the EdgeTAM image encoder (RepViT backbone + FPN neck).
**
** Runs the full graph once for a total timing and op summary, then builds
** and times each stage as a separate sub-graph to produce a per-stage
** latency breakdown:
**   - Stem (2 convolutions)
**   - Stage 0..3 (downsample + RepViT blocks)
**   - FPN neck (lateral convolutions + top-down fusion)
**
** n_warmup iterations are run before n_iter timed iterations.
** Results are printed to stderr.
*/
bool sam3_profile_edgetam_encode(const sam3_model & model,
                                 const sam3_image & image,
                                 int                n_threads = 4,
                                 int                n_warmup  = 2,
                                 int                n_iter    = 5);
