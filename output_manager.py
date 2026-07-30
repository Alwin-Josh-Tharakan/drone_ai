# output_manager.py
"""
Centralised output / dataset writer.

Folder layout (per run)
-----------------------
    <OUTPUT_DIR>/
    └── run_<timestamp>/
        ├── first_qr/
        │   ├── start_qr_raw.png
        │   └── start_qr_payload.txt
        ├── template_qr/
        │   ├── template_raw.png
        │   └── template_processed.png
        ├── candidate_qr/
        │   ├── cand_000123_raw.png
        │   ├── cand_000123_binary.png
        │   └── cand_000123_meta.txt
        ├── logs/        ← used by vision_combined
        ├── planning/    ← used by search_planner.save_plan()
        └── mission_<ts>.avi
"""

from __future__ import annotations
import os
import time
import logging
import json
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OutputManager:
    """Creates the per-run folder tree and provides save helpers."""

    def __init__(self,
                 output_root:            str,
                 timestamp:              Optional[str] = None,
                 candidate_throttle_sec: float = 2.0):
        self.output_root            = output_root
        self.timestamp              = timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S")
        self.candidate_throttle_sec = candidate_throttle_sec

        self.run_dir       = os.path.join(output_root, f"run_{self.timestamp}")
        self.dir_first     = os.path.join(self.run_dir, "first_qr")
        self.dir_template  = os.path.join(self.run_dir, "template_qr")
        self.dir_candidate = os.path.join(self.run_dir, "candidate_qr")
        self.dir_logs      = os.path.join(self.run_dir, "logs")
        self.dir_planning  = os.path.join(self.run_dir, "planning")

        self._last_candidate_save_ts: float = 0.0
        self._candidate_counter:      int   = 0
        self._first_saved:            bool  = False
        self._template_saved:         bool  = False

        self._build_tree()

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    def _build_tree(self) -> None:
        for d in (self.run_dir, self.dir_first, self.dir_template,
                  self.dir_candidate, self.dir_logs, self.dir_planning):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                logger.error(f"[OUT] Could not create {d}: {e}")
        logger.info(f"[OUT] Run dir = {self.run_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # First QR (the START QR scanned by the camera)
    # ──────────────────────────────────────────────────────────────────────

    def save_first_qr(self,
                      raw_crop:  Optional[np.ndarray],
                      payload:   str) -> bool:
        """Persist the raw crop of the START QR + a truncated payload preview."""
        if self._first_saved:
            return False
        try:
            if raw_crop is not None and raw_crop.size > 0:
                path = os.path.join(self.dir_first, "start_qr_raw.png")
                cv2.imwrite(path, raw_crop)
                logger.info(f"[OUT] Saved first QR raw → {path}")

            preview = (payload[:300] + "…[truncated]"
                       if len(payload) > 300 else payload)
            txt_path = os.path.join(self.dir_first, "start_qr_payload.txt")
            with open(txt_path, "w") as f:
                f.write(preview)
            logger.info(f"[OUT] Saved first QR payload → {txt_path}  "
                        f"(len={len(payload)})")

            self._first_saved = True
            return True
        except Exception as e:
            logger.warning(f"[OUT] save_first_qr failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Template QR (the decoded target image rebuilt from the START QR)
    # ──────────────────────────────────────────────────────────────────────

    def save_template(self,
                      raw_image:    Optional[np.ndarray],
                      processed:    Optional[np.ndarray]) -> bool:
        """Persist both the decoded raw target image and its binarised form."""
        if self._template_saved:
            return False
        try:
            if raw_image is not None and raw_image.size > 0:
                p = os.path.join(self.dir_template, "template_raw.png")
                cv2.imwrite(p, raw_image)
                logger.info(f"[OUT] Saved template raw       → {p}")

            if processed is not None and processed.size > 0:
                p = os.path.join(self.dir_template, "template_processed.png")
                cv2.imwrite(p, processed)
                logger.info(f"[OUT] Saved template processed → {p}")

            self._template_saved = True
            return True
        except Exception as e:
            logger.warning(f"[OUT] save_template failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Candidate QR  (every non-START QR seen during the mission, throttled)
    # ──────────────────────────────────────────────────────────────────────

    def save_candidate(self,
                       raw_crop:     Optional[np.ndarray],
                       binary_crop:  Optional[np.ndarray],
                       *,
                       frame_index:  int,
                       confidence:   float = 0.0,
                       matched:      bool  = False,
                       sharpness:    float = 0.0) -> bool:
        """
        Persist one candidate QR + its meta line — but throttled so we don't
        flood the disk.  Returns True if a write happened.
        """
        now = time.time()
        if (now - self._last_candidate_save_ts) < self.candidate_throttle_sec:
            return False

        try:
            self._candidate_counter += 1
            tag = f"cand_{self._candidate_counter:06d}"

            if raw_crop is not None and raw_crop.size > 0:
                p = os.path.join(self.dir_candidate, f"{tag}_raw.png")
                cv2.imwrite(p, raw_crop)

            if binary_crop is not None and binary_crop.size > 0:
                p = os.path.join(self.dir_candidate, f"{tag}_binary.png")
                cv2.imwrite(p, binary_crop)

            meta_path = os.path.join(self.dir_candidate, f"{tag}_meta.txt")
            with open(meta_path, "w") as f:
                json.dump({
                    "tag":          tag,
                    "frame":        frame_index,
                    "timestamp":    datetime.now().isoformat(),
                    "confidence":   round(float(confidence), 4),
                    "matched":      bool(matched),
                    "sharpness":    round(float(sharpness), 2),
                }, f, indent=2)

            self._last_candidate_save_ts = now
            logger.debug(
                f"[OUT] Saved candidate {tag}  conf={confidence:.3f}  "
                f"matched={matched}")
            return True
        except Exception as e:
            logger.warning(f"[OUT] save_candidate failed: {e}")
            return False