#!/bin/bash
# Download test assets for sam3.cpp interactive examples.
# Prerequisites: curl
# Optional: ffmpeg (needed by the video example at runtime)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

mkdir -p "$DATA_DIR"

# ── Test image: COCO val2017 000000039769 (two cats on a couch) ──────────────
IMG="$DATA_DIR/test_image.jpg"
if [ ! -f "$IMG" ]; then
    echo "Downloading test image..."
    curl -L -o "$IMG" "http://images.cocodataset.org/val2017/000000039769.jpg"
    echo "  -> $(ls -lh "$IMG" | awk '{print $5}') downloaded"
else
    echo "Test image already exists: $IMG"
fi

# ── Test video: 5-second sample clip ─────────────────────────────────────────
VID="$DATA_DIR/test_video.mp4"
if [ ! -f "$VID" ]; then
    echo "Downloading test video..."
    curl -L -o "$VID" "https://download.samplelib.com/mp4/sample-5s.mp4"
    echo "  -> $(ls -lh "$VID" | awk '{print $5}') downloaded"
else
    echo "Test video already exists: $VID"
fi

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "Test assets in $DATA_DIR:"
ls -lh "$DATA_DIR"

if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "WARNING: ffmpeg not found. The video example requires ffmpeg at runtime."
    echo "  macOS:  brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
fi
