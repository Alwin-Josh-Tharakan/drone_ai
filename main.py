# main.py — USB Camera Initialization, Calibration & Vision Entry Point
#
# Mission fork objective: replace the Picam3 capture path with the
# SINOSEE global-shutter USB module and optimize the vision pipeline.
# Detector implementations live in detectors/ (same layout as the baseline):
#   detectors/qr_detector.py       — QR location + visual template matching
#   detectors/template_matcher.py  — high-precision binary verification
#   detectors/green_banner.py / red_zone.py / corridor_nav.py — other detectors
#
# This file was previously camera_config.py; renamed to main.py per the new
# workflow (main file holds all camera/main-loop code).

import os

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CAMERA INTRINSICS (Calibration Results)
# Obtained from offline checkerboard calibration of the
# SINOSEE global-shutter USB module @ 1600x1300.
# ═══════════════════════════════════════════════════════════════

# Camera matrix (intrinsic parameters)
CAMERA_MATRIX = np.array([
    [983.77560081, 0.0,            820.43217513],
    [0.0,          983.68738997,   628.51796495],
    [0.0,          0.0,            1.0]
])

# Distortion coefficients (k1, k2, p1, p2, k3)
DIST_COEFFS = np.array([
    [-7.42731947e-02,  8.17471050e-02,  2.31106726e-03,
     -7.08036420e-05, -3.43309770e-02]
])

# Principal point (cx, cy) and mean focal length (px) — convenience
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2.0
PRINCIPAL_POINT = (CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2])


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CAMERA INITIALIZATION (SINOSEE USB, MJPG backend)
# ═══════════════════════════════════════════════════════════════

def _open_backend(camera_id: int, backend) -> cv2.VideoCapture:
    """Open one backend; returns cap only if it actually opened."""
    cap = cv2.VideoCapture(camera_id, backend)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def initialize_camera(camera_id: int = 1,
                      width: int = 1600,
                      height: int = 1300) -> cv2.VideoCapture:
    """
    Initializes and configures the SINOSEE global shutter USB camera
    with optimal settings for detection / calibration / capture.

    Robustness notes (laptop testing):
      * On Linux, a webcam is often /dev/video0 (or another index) — the
        hardcoded index is tried FIRST, then all other indices 0..9.
      * Some builds/devices reject CAP_V4L2 or CAP_DSHOW outright, so we
        fall back to the OS default backend before giving up.

    Args:
        camera_id (int): Preferred OpenCV camera index
                         (pass -1 to skip straight to auto-detect)
        width (int): Desired frame width
        height (int): Desired frame height

    Returns:
        cv2.VideoCapture: Configured camera object

    Raises:
        RuntimeError: If no working camera could be opened.
    """
    # 1. Choose backend based on OS (DirectShow for Windows, V4L2 for Linux),
    #    with the OS-default backend as fallback.
    if os.name == 'nt':
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # 2. Candidate indices: preferred first, then scan 0..9 (skip duplicates).
    preferred = [] if camera_id is None or camera_id < 0 else [camera_id]
    candidates = preferred + [i for i in range(10) if i not in preferred]

    cap = None
    used_idx = None
    for idx in candidates:
        for backend in backends:
            cap = _open_backend(idx, backend)
            if cap is not None:
                used_idx = idx
                break
        if cap is not None:
            if idx != camera_id:
                print(f"[main] Camera at requested ID {camera_id} unavailable — "
                      f"opened index {idx} instead. "
                      f"Set config.CAMERA_ID = {idx} to make this permanent.")
            break
        cap = None

    if cap is None:
        _bname = {cv2.CAP_DSHOW: 'DSHOW', cv2.CAP_V4L2: 'V4L2',
                  getattr(cv2, 'CAP_ANY', -99): 'ANY'}
        _backends_str = "/".join(_bname.get(b, str(b)) for b in backends)
        raise RuntimeError(
            f"Error: Could not open any camera (tried IDs "
            f"{', '.join(map(str, candidates))} with backends {_backends_str}).\n"
            f"Checklist:\n"
            f"  1) ls /dev/video*   — is the module enumerated?\n"
            f"  2) v4l2-ctl --list-devices\n"
            f"  3) sudo usermod -aG video $USER   (then log out/in)\n"
            f"  4) Is another app (e.g. the main.py self-test window) still holding the camera?")
    camera_id = used_idx

    # 3. Set FourCC to MJPG (required for high-resolution/high-FPS over USB)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 4. Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 5. Configure Exposure and White Balance
    # Note: Windows DirectShow uses 0.75 for Manual Exposure, 0.25 for Auto.
    # Linux V4L2 uses 1 for Manual, 3 for Auto (Aperture Priority).
    auto_exposure_val = 0.75 if os.name == 'nt' else 3
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_val)

    # 1 = Auto White Balance, 0 = Manual
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    # 6. Verify actual resolution (cameras often fallback to closest supported mode)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[main] Initialized successfully. "
          f"Actual resolution: {actual_width}x{actual_height}")

    return cap


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DISTORTION CORRECTION HELPERS
# ═══════════════════════════════════════════════════════════════

def undistort_frame(frame: np.ndarray) -> np.ndarray:
    """Apply full distortion correction to a raw frame."""
    return cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)


def get_undistort_maps(width: int, height: int):
    """
    Precompute rectify maps once for a fixed resolution, then reuse
    with cv2.remap() per-frame (much faster than cv2.undistort in a loop).

    IMPORTANT: the intrinsics above were calibrated at 1600x1300. If the
    camera negotiated a different resolution (common on laptops where the
    full sensor mode may be unavailable), the matrix is scaled accordingly
    so undistortion stays valid; if no calibration applies to the actual
    size, callers should treat maps as identity (no-op remap).

    Returns:
        (map1, map2, new_matrix, roi) — same order as before, but also
        usable as `map1, map2 = get_undistort_maps(w, h)[:2]`.
    """
    cal_w = float(CAMERA_MATRIX[0, 2] * 2.0)   # ~2*cx ≈ calibrated width
    cal_h = float(CAMERA_MATRIX[1, 2] * 2.0)   # ~2*cy ≈ calibrated height
    sx, sy = width / cal_w, height / cal_h

    if abs(sx - 1.0) > 0.05 or abs(sy - 1.0) > 0.05:
        print(f"[main] WARNING: resolution {width}x{height} differs from "
              f"calibration (~{int(cal_w)}x{int(cal_h)}); scaling intrinsics "
              f"by ({sx:.2f}, {sy:.2f}). Re-calibrate at this resolution "
              f"for best accuracy.")
        K = CAMERA_MATRIX.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    else:
        K = CAMERA_MATRIX

    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        K, DIST_COEFFS, (width, height), alpha=1,
        newImgSize=(width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        K, DIST_COEFFS, None, new_matrix,
        (width, height), cv2.CV_32FC1
    )
    return map1, map2, new_matrix, roi


def pixel_to_bearing(x: float, y: float, frame_size=(1600, 1300)):
    """
    Convert an undistorted pixel coordinate to a horizontal/vertical
    bearing angle (degrees) relative to the optical axis.
    Useful for turning detector offsets into yaw/pitch commands.
    """
    cx, cy = PRINCIPAL_POINT
    dx = x - cx
    dy = y - cy
    yaw_deg = np.degrees(np.arctan2(dx, FOCAL_LENGTH))
    pitch_deg = np.degrees(np.arctan2(dy, FOCAL_LENGTH))
    return yaw_deg, pitch_deg


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — AUTO-DETECT CAMERA & SELF-TEST
# ═══════════════════════════════════════════════════════════════

def find_camera(width: int = 1600, height: int = 1300):
    """
    Auto-detect a working camera. initialize_camera() already scans all
    indices (preferred first, then 0..9) and both backends, so this is a
    thin wrapper kept for API compatibility.

    Returns:
        cv2.VideoCapture
    """
    return initialize_camera(camera_id=-1, width=width, height=height)


if __name__ == "__main__":
    try:
        # Auto-detect: tries every index/backend until one streams
        cap = find_camera(width=1600, height=1300)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        map1, map2, _, _ = get_undistort_maps(w, h)

        print("Press 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Real-time undistortion using precomputed maps
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            cv2.imshow("SINOSEE USB - Undistorted", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()