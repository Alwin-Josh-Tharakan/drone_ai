#!/usr/bin/env python3
"""
main.py — GStreamer UDP Receiver & Calibration Constants
Replaces local USB camera init with a network stream receiver from the Pi.
"""
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CAMERA INTRINSICS (Calibration Results)
# SINOSEE global-shutter USB module @ 1600x1200 (Native Res)
# Global RMS Error: 0.2787 px
# ═══════════════════════════════════════════════════════════════
CAMERA_MATRIX = np.array([
    [983.8432,   0.0000,   820.4136],
    [0.0000,     983.7728, 628.5018],
    [0.0000,     0.0000,   1.0000]
])

DIST_COEFFS = np.array([
    [-0.074344,  0.081757,  0.002323, 
     -0.000085, -0.034161]
])

FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2.0
PRINCIPAL_POINT = (CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2])


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — GSTREAMER CAMERA INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def initialize_camera(gst_port: int = 5000) -> cv2.VideoCapture:
    """
    Initializes the GStreamer UDP receiver to get video from the Raspberry Pi.
    
    Args:
        gst_port (int): UDP port to listen on (default: 5000)

    Returns:
        cv2.VideoCapture: Opened GStreamer stream object
    """
    # GStreamer pipeline to receive MJPEG over UDP
    gst_pipeline = (
        f"udpsrc port={gst_port} "
        f"caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG\" ! "
        f"rtpjpegdepay ! jpegdec ! videoconvert ! "
        f"appsink drop=1 sync=false"
    )

    print(f"[main] Opening GStreamer receiver on port {gst_port}...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open GStreamer stream on port {gst_port}. Is the Pi streaming?")

    # Read one frame to force the pipeline to negotiate resolution
    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("GStreamer stream opened, but failed to read the first frame.")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[main] GStreamer stream initialized successfully.")
    print(f"[main] Actual resolution: {actual_width}x{actual_height}")

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
    """
    cx, cy = PRINCIPAL_POINT
    dx = x - cx
    dy = y - cy
    yaw_deg = np.degrees(np.arctan2(dx, FOCAL_LENGTH))
    pitch_deg = np.degrees(np.arctan2(dy, FOCAL_LENGTH))
    return yaw_deg, pitch_deg


if __name__ == "__main__":
    try:
        cap = initialize_camera(gst_port=5000)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        map1, map2, _, _ = get_undistort_maps(w, h)

        print("Press 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            cv2.imshow("GStreamer Receiver - Undistorted", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()