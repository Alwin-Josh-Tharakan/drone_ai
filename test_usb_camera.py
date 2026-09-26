"""
Laptop test harness for the SinoSeen USB global-shutter module.
Tests end-to-end: camera init -> undistort maps -> live preview ->
QR detection (pyzbar) -> template verification (TemplateMatcher).

Usage:
    python test_usb_camera.py             # tries camera index 1, then scans 0..3
    python test_usb_camera.py --camera 0  # force a specific index
Keys: ESC = quit

Offline smoke-test (no camera needed):
    python test_usb_camera.py --selftest
"""
import argparse
import base64
import time

import cv2
import numpy as np

from main import initialize_camera, get_undistort_maps
from detectors.qr_detector import QRDetector
from detectors.template_matcher import TemplateMatcher


def find_camera():
    """Try camera indices from 4 down to 0."""
    for idx in range(4, -1, -1):
        try:
            cap = initialize_camera(camera_id=idx)
            print(f"[OK] Camera found at index {idx}")
            return cap, idx
        except RuntimeError:
            print(f"[..] No camera at index {idx}, trying next...")
    raise RuntimeError("No USB camera found on indices 4 down to 0. Check cable/permissions.")

def render_qr_png(data: str) -> np.ndarray:
    """Render *data* as a crisp black/white QR image (numpy array)."""
    from qrcode import QRCode, ERROR_CORRECT_M  # pip install qrcode[pil]
    qc = QRCode(error_correction=ERROR_CORRECT_M)
    qc.add_data(data)
    pil = qc.make_image().get_image(box_size=8, border=2).convert("L")
    return np.array(pil)


def make_synthetic_qr_payload() -> str:
    """Simulate what the START QR carries: a base64 PNG of the target image.
    The target itself is a scannable QR encoding 'TARGET-42' — realistic and
    high-contrast, so both the coarse grid match and fine matcher can verify."""
    target = render_qr_png("TARGET-42")
    ok, buf = cv2.imencode(".png", target, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return base64.b64encode(buf.tobytes()).decode()


def selftest():
    """Run the full detector pipeline on synthetic frames, no camera needed."""
    print("=== OFFLINE SELFTEST ===")
    qr_det = QRDetector()
    matcher = TemplateMatcher(frame_size=(1600, 1300))

    payload = make_synthetic_qr_payload()
    td = qr_det.load_target_from_qr_payload(payload)
    assert td is not None and td.template_qr_image is not None, "template load failed"
    print("[OK] Target template loaded from simulated QR payload")

    # Render the TARGET image itself as a physical QR to be found in-frame
    qr_img = render_qr_png("TARGET-42")

    frame = np.full((1300, 1600, 3), 128, dtype=np.uint8)
    h, w = qr_img.shape
    y0, x0 = (1300 - h) // 2, (1600 - w) // 2
    frame[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)

    t0 = time.perf_counter()
    qrs = qr_det.detect(frame)
    dt = (time.perf_counter() - t0) * 1000
    assert len(qrs) == 1, f"expected 1 QR, got {len(qrs)}"
    data, (cx, cy), rect, raw_crop, bin_crop = qrs[0]
    print(f"[OK] detect(): 1 QR @ ({cx},{cy}) rect={rect} in {dt:.1f} ms")

    ok_flag, conf = qr_det.match_against_target(bin_crop)
    print(f"[OK] match_against_target(): conf={conf} -> {ok_flag}")

    res = matcher.match(raw_crop, td.template_qr_image, candidate_rect=rect)
    print(f"[{'OK' if res.match_found else 'FAIL'}] TemplateMatcher: "
          f"matched={res.match_found} conf={res.confidence:.3f} dev=({res.deviation_x},{res.deviation_y})")
    assert res.match_found, "template match should succeed on synthetic QR"
    print("=== SELFTEST PASSED ===")


def main():
    ap = argparse.ArgumentParser()
    # Removed --camera argument since we now auto-scan 4 down to 0
    ap.add_argument("--no-undistort", action="store_true", help="skip lens undistortion")
    ap.add_argument("--selftest", action="store_true", help="offline test, no camera")
    ap.add_argument("--camera", type=int, default=None,
                    help="preferred camera index (default: auto-scan all indices)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    # ---- 1. Camera ----
    # initialize_camera() now auto-scans indices/backends internally, so a
    # single call with the CLI index (or -1 to skip straight to scanning) works.
    cap = initialize_camera(camera_id=args.camera if args.camera is not None else -1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- 2. Undistort maps (precomputed once — remap is ~5x faster than undistort) ----
    if args.no_undistort:
        map1 = map2 = None
    else:
        map1, map2, _new_matrix, roi = get_undistort_maps(w, h)
        print(f"[OK] Undistort maps built for {w}x{h}, ROI={roi}")

    # ---- 3. Detectors ----
    qr_det = QRDetector()
    matcher = TemplateMatcher(frame_size=(w, h))
    print("Waiting for START QR to load target template... "
          "(hold the start QR in front of the camera)")

    print("Press ESC to exit.")
    fps_t, n = time.time(), 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed (device may have disconnected)")
            break

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        qrs = qr_det.detect(frame)

        # First QR seen = START QR -> load its payload as the target template
        if qrs and qr_det.get_target_template() is None:
            td = qr_det.load_target_from_qr_payload(qrs[0][0])
            if td is not None:
                print(f"[OK] Target template loaded from live START QR "
                      f"(payload {len(qrs[0][0])} chars)")

        for data, (cx, cy), rect, raw_crop, bin_crop in qrs:
            x, y, rw, rh = rect
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

            if qr_det.get_target_template() is not None:
                ok_flag, conf = qr_det.match_against_target(bin_crop)
                res = matcher.match(raw_crop, qr_det.get_target_template(),
                                    candidate_rect=rect)
                label = ("MATCH %.2f/%.2f" % (conf, res.confidence)
                         if res.match_found else "no-match %.2f" % res.confidence)
                color = (0, 255, 0) if res.match_found else (0, 0, 255)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if res.match_found:
                    cv2.putText(frame,
                                f"dev=({res.dev_x:+d},{res.dev_y:+d})",
                                (x, y + rh + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        n += 1
        if time.time() - fps_t >= 1.0:
            print(f"[FPS] {n / (time.time() - fps_t):.1f} | QRs visible: {len(qrs)}")
            fps_t, n = time.time(), 0

        # Downscale preview so 1600x1300 fits on a laptop screen
        prev = cv2.resize(frame, (w // 2, h // 2))
        cv2.imshow("USB Camera Test (SinoSeen)", prev)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
