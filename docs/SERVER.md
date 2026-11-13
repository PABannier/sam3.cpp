# sam3-server

HTTP-сервер в стиле llama-server для sam3.cpp (в стиле llama.cpp `llama-server`).
Сборка: `cmake --build build --target sam3-server`. Зависимости: cpp-httplib и
nlohmann/json (vendored single-header в `examples/sam3-server/`), ffmpeg/ffprobe
для видео (декодирование кадров через `popen ffmpeg`).

## Запуск

```bash
./build/examples/sam3-server -m sam3-f16 --models-dir models \
    --host 0.0.0.0 --port 8080 --threads 32 --gpu --slots 2
```

| Флаг | Описание |
|---|---|
| `-m, --model NAME` | файл модели или имя в `--models-dir` (добавляется `.ggml`) |
| `--models-dir DIR` | каталог моделей |
| `--host HOST` | интерфейс (по умолчанию `127.0.0.1`) |
| `--port PORT` | порт (по умолчанию `8080`) |
| `-t, --threads N` | потоки CPU |
| `--gpu` / `--nogpu` | использовать GPU (HIP/CUDA) |
| `--gpu-device N` | индекс устройства |
| `--encode-img-size N` | переопределить входное разрешение изображения |
| `--slots N` | число параллельных слотов |
| `--score-threshold F` | дефолтный score-порог детекции (по умолчанию `0.5`) |
| `--nms-threshold F` | дефолтный IoU-порог NMS (по умолчанию `0.1`) |
| `--mask-min-area N` | cleanup: удалять фрагменты маски ≤ N px (по умолчанию `64`, `0` = выкл) |
| `--mask-fill-hole-area N` | cleanup: заполнять замкнутые дыры ≤ N px (по умолчанию `2048`, `0` = выкл) |
| `--no-webui` | отключить веб-интерфейс |

## Параметры детекции и трекинга

| Параметр | Дефолт | CLI | Тело запроса | Где действует | Смысл |
|---|---|---|---|---|---|
| `score_threshold` | `0.5` | `--score-threshold` | `/detect`, `/video/track` | детекция PCS | минимальный score детекции DETR |
| `nms_threshold` | `0.1` | `--nms-threshold` | `/detect`, `/video/track` | детекция PCS | IoU-порог подавления перекрывающихся боксов (NMS) |
| `assoc_iou_threshold` | `0.1` | — | `/video/track` | видео-трекинг | IoU-порог ассоциации треков с детекциями |
| `hotstart_delay` | `0` (в видео-эндпоинте) | — | `/video/track` | видео-трекинг | кадры до начала детекции |
| `max_keep_alive` | `100` | — | `/video/track` | видео-трекинг | кадры жизни трека без детекции |
| `recondition_every` | `16` | — | `/video/track` | видео-трекинг | периодичность рекондиционирования трека |
| `fill_hole_area` | `16` | — | `/video/track` | видео-трекинг | минимальная площадь дырки в маске |

Приоритет для `/detect` и `/video/track`: значение из тела запроса, если оно
есть, иначе серверный дефолт (CLI-флаг). У `sam3_pcs_params`/`sam3_video_params`
и в веб-UI поля те же.

> **Урок кейса case4 (три человека вплотную):** при `nms_threshold=0.1`
> (дефолт) боксы плотно стоящих объектов подавляют друг друга — модель
> детектит только крайних, средняя (score 0.92) теряется. Для сцен с
> перекрывающимися боксами задавайте `nms_threshold ≥ 0.5` в запросе,
> CLI-флаге или поле «nms» веб-UI.

## Эндпоинты

### Общие
- `GET /health` — статус: `{status, model, visual_only, backend, slots, slots_busy}`.
- `GET /models` — список `*.ggml` в `--models-dir` (через `ls`).
- `POST /model/load` — переключение модели на лету: `{"model": "sam3-f16"}` (имя из
  `--models-dir` без `.ggml` или путь). Ждёт освобождения всех слотов, грузит модель,
  пересоздаёт стейты слотов и очищает сессии трекинга. Ответ:
  `{model, visual_only, backend}`; при ошибке старый модель остаётся активной.
- `GET /` — веб-интерфейс (canvas: точки/бокс, оверлей масок, видео-плеер).

### Изображения
- `POST /segment` — PVS-сегментация. Тело: `{image_path | image_base64, points:[{x,y,label:1|-1}], box:{x0,y0,x1,y1}, multimask}`.
  Ответ: `{masks:[{width,height,iou_score,obj_score,instance_id,mask_base64}], time_ms}`.
- `POST /detect` — PCS-детекция (только SAM3, не SAM2/visual-only). Тело:
  `{image_path | image_base64, text_prompt, exemplars:[{box:{x0,y0,x1,y1}, label:"pos"|"neg"}], score_threshold, nms_threshold}`.
  Ответ: `{detections:[{box,score,iou_score,instance_id,mask}], time_ms}`.

### Очистка масок (cleanup)
`/detect` и `/segment` поддерживают opt-in постобработку бинарных масок:
`"cleanup": true` в теле запроса. Выполняется после инференса, в ответе маски уже
очищены:
- удаляются фрагменты переднего плана с площадью ≤ `min_area` (спёклы/оторванные точки);
- заполняются замкнутые дыры с площадью ≤ `fill_hole_area` (BFS; дыры, касающиеся
  края изображения, не трогаются).

Пороги: `"min_area"`/`"fill_hole_area"` в теле (приоритет) → CLI-дефолты
`--mask-min-area` (64) / `--mask-fill-hole-area` (2048); `0` отключает стадию.
В веб-UI — чекбокс «cleanup mask» (пороги берутся с сервера).

Пример: `{"image_path": "frame.png", "text_prompt": "woman", "cleanup": true,
"min_area": 64, "fill_hole_area": 2048}`. Слишком большой `min_area` может
удалить настоящие части объекта (например, отдельно стоящий человек — фрагмент
в тысячи px), а `fill_hole_area` — заполнить осмысленные отверстия; подбирайте
пороги под сцену.

### Трекинг (по кадрам-изображениям)
- `POST /track/new` — создать сессию, `{session_id}`.
- `POST /track/instances` — `{session_id, image_path|image_base64, points, box, multimask}` → `{instance_id}`.
- `POST /track/frame` — `{session_id, image_path|image_base64}` → `{frame_index, detections}`.
- `POST /track/reset` — `{session_id}`.

### Видео
- `POST /video/upload` — multipart поле `file` → `{video_id, width, height, n_frames, fps}`.
- `POST /video/open` — `{url}`: http(s) (скачивается в `/tmp/sam3-server-videos/`),
  путь к локальному файлу, либо `rtsp://`/`rtmp://`/`udp://` (ffmpeg читает поток напрямую).
  Ответ: `{video_id, width, height, n_frames, fps, live}` (`live=true` у потоков).
- `GET /video/info?video_id=` — метаданные.
- `GET /video/frame?video_id=&frame=N` — кадр как PNG.
- `POST /video/track` — трек по всему видео: `{video_id, points, box, text_prompt, max_frames}`
  или мульти-инстанс: `{video_id, instances: [{points, box}, ...], text_prompt?, max_frames}` —
  каждый элемент `instances` становится отдельным трекинг-инстансом (своя маска и
  instance_id на кадре); `text_prompt` (SAM3) дополнительно включает PCS-детекцию
  на каждом кадре с автоассоциацией новых объектов по `assoc_iou_threshold`.
  Ответ: `{video_id, frames: [{index, detections: [{box, score, iou_score,
  instance_id, mask}]}], time_ms}`; прогресс кадров — в stderr-логе сервера.
  Ответ: `{video_id, frames:[{index, detections}], time_ms}`.
  Для живых потоков `max_frames` обязателен (индексация кадров по порядку чтения).

`mask_base64` — base64 **сырых** байтов бинарной маски (0/255) размером
`width*height`, маска в разрешении исходного изображения.

## Примеры

```bash
# health
curl http://127.0.0.1:8080/health

# сегментация по файлу
curl -s -X POST http://127.0.0.1:8080/segment -H 'Content-Type: application/json' \
  -d '{"image_path":"/path/frame.png","points":[{"x":315,"y":250}]}'

# детекция текстом + exemplar-бокс
curl -s -X POST http://127.0.0.1:8080/detect -H 'Content-Type: application/json' \
  -d '{"image_path":"/path/frame.png","text_prompt":"a cat",
       "exemplars":[{"box":{"x0":200,"y0":100,"x1":400,"y1":300},"label":"pos"}]}'

# видео: upload + track
curl -s -F 'file=@video.mp4' http://127.0.0.1:8080/video/upload
curl -s -X POST http://127.0.0.1:8080/video/track -H 'Content-Type: application/json' \
  -d '{"video_id":"v1","points":[{"x":315,"y":250}],"max_frames":10}'

# видео: поток
curl -s -X POST http://127.0.0.1:8080/video/open -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://cam.local/stream1"}'
```

```python
import base64, json, urllib.request

def post(path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:8080{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

r = post("/segment", {
    "image_path": "/path/frame.png",
    "points": [{"x": 315, "y": 250, "label": 1}],
})
mask = base64.b64decode(r["masks"][0]["mask_base64"])  # bytes, 0/255
w, h = r["masks"][0]["width"], r["masks"][0]["height"]
```

## Ограничения

- Один слот = один `sam3_state` (энкодер ViT ~2500 узлов) — параллелизм ограничен
  числом слотов и памятью (SAM3 f16 на GPU: ~1.7GB веса + активации).
- q4_1 на GPU поддерживается в новой базе ggml (strix-llama, MMQ RDNA3.5);
  в старой PABannier-базе — только CPU.
- Для живых потоков ffmpeg-декодер читает кадры последовательно; «секунды»
  соответствуют `frame / fps`, покадровый seek нестабилен.
- Веб-UI использует fetch на тот же origin (same-origin, CORS не требуется).
