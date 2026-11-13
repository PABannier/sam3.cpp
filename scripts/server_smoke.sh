#!/usr/bin/env bash
# Smoke test for sam3-server (CPU, no GPU required).
# Usage: bash scripts/server_smoke.sh [path_to_sam3-server] [model.ggml] [image] [video.mp4]
set -euo pipefail

SERVER="${1:-build/examples/sam3-server}"
MODEL="${2:-models/sam2.1_hiera_tiny_f16.ggml}"
IMAGE="${3:-/tmp/opencode/frame0.png}"
VIDEO="${4:-}"
PORT=8123
BASE="http://127.0.0.1:${PORT}"

if [ ! -x "$SERVER" ]; then
    echo "error: server binary not found: $SERVER"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL"
    exit 1
fi
if [ ! -f "$IMAGE" ]; then
    echo "error: image not found: $IMAGE"
    exit 1
fi

echo "== starting server on :$PORT"
"$SERVER" --model "$MODEL" --models-dir "$(dirname "$MODEL")" \
    --host 127.0.0.1 --port "$PORT" --nogpu --threads 4 --slots 2 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if curl -fsS "$BASE/health" > /dev/null 2>&1; then break; fi
    sleep 1
done

echo "== /health"
curl -fsS "$BASE/health" | python3 -m json.tool

echo "== /models"
curl -fsS "$BASE/models" | python3 -m json.tool

echo "== /segment (point)"
curl -fsS -X POST "$BASE/segment" \
    -H 'Content-Type: application/json' \
    -d "{\"image_path\": \"$IMAGE\", \"points\": [{\"x\": 315, \"y\": 250, \"label\": 1}]}" \
    | python3 -c 'import json,sys; j=json.load(sys.stdin); print("masks:", len(j["masks"]), "time_ms:", round(j["time_ms"],1))'

echo "== /track"
TID=$(curl -fsS -X POST "$BASE/track/new" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "session: $TID"
curl -fsS -X POST "$BASE/track/instances" \
    -H 'Content-Type: application/json' \
    -d "{\"id\": \"$TID\", \"image_path\": \"$IMAGE\", \"points\": [{\"x\": 315, \"y\": 250, \"label\": 1}]}" \
    | python3 -m json.tool
curl -fsS -X POST "$BASE/track/frame" \
    -H 'Content-Type: application/json' \
    -d "{\"id\": \"$TID\", \"image_path\": \"$IMAGE\"}" \
    | python3 -c 'import json,sys; j=json.load(sys.stdin); print("frame:", j["frame_index"], "dets:", len(j["detections"]))'
curl -fsS -X POST "$BASE/track/reset" \
    -H 'Content-Type: application/json' \
    -d "{\"id\": \"$TID\"}" | python3 -m json.tool

echo "== / (web UI)"
curl -fsS "$BASE/" | head -c 120; echo

if [ -n "$VIDEO" ] && [ -f "$VIDEO" ]; then
    echo "== /video/upload + /video/track (async job)"
    VID=$(curl -fsS -X POST "$BASE/video/upload" -F "file=@$VIDEO" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["video_id"])')
    echo "video: $VID"
    INFO=$(curl -fsS "$BASE/video/info?video_id=$VID")
    W=$(python3 -c "import json,sys; print(json.load(sys.stdin)['width'])" <<< "$INFO")
    H=$(python3 -c "import json,sys; print(json.load(sys.stdin)['height'])" <<< "$INFO")
    X=$((W / 2)); Y=$((H / 2))
    JOB=$(curl -fsS -X POST "$BASE/video/track" \
        -H 'Content-Type: application/json' \
        -d "{\"video_id\": \"$VID\", \"max_frames\": 2, \"points\": [{\"x\": $X, \"y\": $Y, \"label\": 1}]}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
    echo "job: $JOB"
    for i in $(seq 1 120); do
        ST=$(curl -fsS "$BASE/job/$JOB" | python3 -c 'import json,sys; j=json.load(sys.stdin); print(j["status"] + " " + str(j.get("progress",{}).get("frames_done",0)) + "/" + str(j.get("progress",{}).get("n_frames",0)))')
        echo "poll: $ST"
        case "$ST" in done*|error*|cancelled*) break;; esac
        sleep 2
    done
    curl -fsS "$BASE/job/$JOB" | python3 -c 'import json,sys; j=json.load(sys.stdin); print("job status:", j["status"]); assert j["status"] == "done", j; print("frames:", len(j["result"]["frames"])); assert len(j["result"]["frames"]) == 2'
    echo "== /job cancel flow"
    JOB2=$(curl -fsS -X POST "$BASE/video/track" \
        -H 'Content-Type: application/json' \
        -d "{\"video_id\": \"$VID\", \"max_frames\": 100, \"points\": [{\"x\": $X, \"y\": $Y, \"label\": 1}]}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
    curl -fsS -X POST "$BASE/job/$JOB2/cancel" | python3 -m json.tool
    for i in $(seq 1 30); do
        ST=$(curl -fsS "$BASE/job/$JOB2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
        case "$ST" in running) sleep 2;; *) break;; esac
    done
    curl -fsS "$BASE/job/$JOB2" | python3 -c 'import json,sys; j=json.load(sys.stdin); print("cancelled job status:", j["status"]); assert j["status"] in ("cancelled", "done", "error"), j'
fi

echo "== SMOKE OK"
