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

def initialize_camera(camera_id: int = 1,
                      width: int = 1600,
                      height: int = 1300) -> cv2.VideoCapture:
    """
    Initializes and configures the SINOSEE global shutter USB camera
    with optimal settings for detection / calibration / capture.

    Args:
        camera_id (int): OpenCV camera index (0 = default, 1 = external USB, ...)
        width (int): Desired frame width
        height (int): Desired frame height

    Returns:
        cv2.VideoCapture: Configured camera object
    """
    # 1. Choose backend based on OS (DirectShow for Windows, V4L2 for Linux)
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2

    # 2. Initialize capture
    cap = cv2.VideoCapture(camera_id, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open camera with ID {camera_id}.")

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

    Usage:
        map1, map2 = get_undistort_maps(1600, 1300)
        clean = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
    """
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        CAMERA_MATRIX, DIST_COEFFS, (width, height), alpha=1,
        newImgSize=(width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX, DIST_COEFFS, None, new_matrix,
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
# Example Usage / Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        cap = initialize_camera(camera_id=1, width=1600, height=1300)
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
        print(e)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
