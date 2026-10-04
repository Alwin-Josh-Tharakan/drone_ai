# drone_ai
sae drone ai and its connection to pixhawk
# drone_ai — QR Detection & Visual Template Matching (Module Pack)

Purpose: locate QR codes in live camera frames and verify — purely by
visual comparison, never by decoded text — whether a detected QR is the
mission TARGET QR (delivered as a base64 PNG inside the START QR payload).

---
## 1. Contents of this pack
---
REQUIRED (core engine, no internal dependencies):
  detectors/__init__.py          package marker (empty)
  detectors/qr_detector.py       QR location (pyzbar) + module-grid visual match
  detectors/template_matcher.py  high-precision binary template verification

RECOMMENDED (support):
  main.py            camera init + calibration + undistort maps
  config.py          tunable thresholds/sizes
  requirements.txt   third-party deps
  test_usb_camera.py offline self-test + live preview (optional)

NOT included (mission-level, not needed for QR detection/matching):
  vision_combined.py, mission_logic.py, output_manager.py,
  planning/, detectors/green_banner.py, red_zone.py, corridor_nav.py

---
## 2. Dependencies
---
Python >= 3.9
  pip:    opencv-python, numpy, pyzbar
  Linux:  sudo apt install libzbar0        (system lib required by pyzbar)
  tests:  qrcode[pil]                      (only for synthetic self-test)

---
## 3. Module APIs — inputs and outputs
---

### 3.1 detectors/qr_detector.py — class QRDetector

Constructor inputs:
  min_size      int    min QR side in px to accept            (default 50)
  output_dir    str    folder for internal debug saves
  template_size tuple  binary template size                   (default (128,128))
  grid_n        int    module grid used for comparison        (default 29)
  detect_scale  float  pyzbar downscale factor for speed      (default 0.5)

Methods:
  load_target_from_qr_payload(qr_payload: str) -> TargetData | None
    IN : text decoded from START QR — base64 PNG; accepts raw base64,
         data-URI ("data:image/png;base64,...") or prefixes IMG:/TARGET:/TPL:/B64:
    OUT: TargetData(template_qr_image, template_qr_raw, timestamp,
         template_filename, payload_size)  or None on failure

  detect(frame: np.ndarray) -> list[tuple]
    IN : full frame, BGR uint8 (H,W,3)
    OUT: one tuple per QR:
         (data_str, (cx,cy), (x,y,w,h), raw_crop, binary_crop)
         all coordinates in FULL-FRAME pixels; binary_crop = uint8 0/255

  match_against_target(binary_crop, threshold=0.80) -> (bool, float)
    IN : binary crop from detect()
    OUT: (match_found, confidence in [0,1]) — best over rotations 0/90/180/270

  draw(frame, detections) -> None
    IN/OUT: draws boxes + confidence HUD on frame in place

### 3.2 detectors/template_matcher.py — class TemplateMatcher

Constructor inputs:
  match_threshold  float  accept threshold                    (default 0.75)
  template_size    tuple  (128,128)
  clahe_limit      float  2.0
  clahe_tile_size  tuple  (8,8)
  frame_size       tuple  (W,H) — used for deviation-from-centre output

Methods:
  process_to_binary(img) -> np.ndarray | None
    IN : BGR or gray image, any size
    OUT: binary uint8 same size, values 0/255 (CLAHE + adaptive threshold)

  preprocess_candidate(frame, rect=None) -> np.ndarray | None
    IN : full frame + optional (x,y,w,h); applies 8% margin crop
    OUT: BGR crop (binarisation happens later inside match())

  match(candidate_frame, template_binary, candidate_rect=None) -> MatchResult
    IN : raw crop, stored binary template, optional rect for deviation
    OUT: MatchResult(match_found, confidence, center, angle,
                     deviation_x, deviation_y)

### 3.3 main.py — camera & calibration support
  initialize_camera(camera_id, width=1600, height=1300) -> cv2.VideoCapture
    OUT: opened cap (MJPG, auto-exposure/WB set); raises RuntimeError if absent
  get_undistort_maps(w, h) -> (map1, map2, new_camera_matrix, roi)
    use per frame:  clean = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
  undistort_frame(frame) -> undistorted frame
  pixel_to_bearing(x, y) -> (yaw_deg, pitch_deg)
  Constants: CAMERA_MATRIX, DIST_COEFFS
    (Charuco calibration @1600x1300, global RMS 0.2792 px)

### 3.4 config.py — key knobs (if shared)
  CAMERA_ID / CAMERA_WIDTH / CAMERA_HEIGHT, QR_MIN_SIZE, TEMPLATE_SIZE,
  MATCH_THRESHOLD, SHARPNESS_THRESHOLD, CLAHE_LIMIT, CLAHE_TILE_SIZE,
  RESOLUTION, FPS, SAVE_FIRST_QR / SAVE_TEMPLATE_QR / SAVE_CANDIDATES

---
## 4. Data-format contract
---
Frame input     : np.ndarray uint8 BGR, shape (H, W, 3)
rect            : (x, y, w, h) ints, full-frame pixels
binary images   : uint8 containing ONLY 0 and 255
template        : binary uint8 at template_size (128x128)
START payload   : str, base64-encoded PNG of the TARGET QR image
confidence      : float 0.0 – 1.0 (higher = more likely same physical QR)

---
## 5. Minimal usage example
---
```python
import cv2
from detectors.qr_detector import QRDetector
from detectors.template_matcher import TemplateMatcher

qr = QRDetector(min_size=50, output_dir="./output")
tm = TemplateMatcher(frame_size=(1600, 1300))

cap = cv2.VideoCapture(0)
template = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    dets = qr.detect(frame)

    # First QR seen = START QR -> build template from its payload
    if template is None and dets:
        td = qr.load_target_from_qr_payload(dets[0][0])
        if td is not None:
            template = td.template_qr_image

    for data, (cx, cy), rect, raw_crop, bin_crop in dets:
        found, conf = qr.match_against_target(bin_crop)          # coarse
        res = None
        if template is not None:
            res = tm.match(raw_crop, template, candidate_rect=rect)  # fine
        print(data[:20], conf, res.confidence if res else "-")

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
```

---
## 6. Files produced at runtime
---
<output_dir>/mission_data/target_templates/
    target_raw_<ts>.png, target_proc_<ts>.png, metadata.log
(per-run mission tree, only when using mission-level scripts)
    run_<ts>/first_qr/, run_<ts>/template_qr/, run_<ts>/candidate_qr/,
    run_<ts>/logs/, run_<ts>/planning/, run_<ts>/mission_<ts>.avi