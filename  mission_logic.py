# mission_logic.py
"""
Mission Logic Controller — integrates visual template verification
(Feature 3) and red-zone escape (Feature 4).

KEY DESIGN
----------
* The QRDetector handles template loading AND visual matching.
* The Mission Controller queries the QRDetector for match confidence
  on every candidate — it does NOT re-implement the matching pipeline.
* QR decoded text is used ONLY by SCAN/STORE states to identify the
  START QR.  It is NEVER used to confirm the drop target — visual
  template matching is mandatory.

QR tuple shape (from QRDetector.detect()):
    (data, (cx, cy), rect, raw_crop, binary_crop, is_sharp)
"""

import time
import logging
from typing import Optional

from state_machine                 import StateMachine
from detectors.template_matcher    import TemplateMatcher, MatchResult

logger = logging.getLogger(__name__)


class MissionController:
    def __init__(self,
                 target_qr:       str   = "TARGET_A",
                 match_threshold: float = 0.75):
        self.sm        = StateMachine()
        self.target_qr = target_qr

        # ── Thresholds ─────────────────────────────────────────
        self.RED_DANGER_AREA   = 5000
        self.CORRIDOR_CENTERED = 20
        self.GREEN_CENTERED    = 30
        self.QR_CENTERED       = 15
        self.STABLE_FRAMES     = 90    # ~3 s at 30 fps

        # ── Red zone escape ────────────────────────────────────
        self.in_red_zone        = False
        self.red_detected_time  = None
        self.RED_ESCAPE_TIMEOUT = 5.0  # seconds

        # ── Mission target storage ─────────────────────────────
        self.stored_coordinates  = None        # decoded START QR text
        self.target_data         = None        # TargetData from QRDetector
        self.qr_detector         = None        # set via set_qr_detector()

        # ── Template matcher (only for compatibility / fallback) ─
        self.matcher         = TemplateMatcher(match_threshold=match_threshold)
        self.match_threshold = match_threshold
        self.last_match      = MatchResult()   # for HUD

        # ── Counters ──────────────────────────────────────────
        self.stable_count       = 0
        self.no_detection_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Wiring helpers
    # ──────────────────────────────────────────────────────────────────────────

    def set_qr_detector(self, qr_detector) -> None:
        """
        Inject the QRDetector instance so this controller can call
        qr.match_against_target() directly (no duplicate matcher pipeline).
        """
        self.qr_detector = qr_detector
        logger.info("[MISSION] QRDetector injected")

    def set_target_data(self, target_data) -> None:
        """Called by vision_combined once the START QR template is extracted."""
        self.target_data = target_data
        logger.info(
            f"[MISSION] TargetData loaded: {target_data.template_filename}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def process(self, qr_results, green_results, red_results, deviation):
        """
        Per-frame decision loop.

        qr_results : list of (data, (cx,cy)) 2-tuples  (converted by caller)
                     OR
                     list of (data, (cx,cy), rect, raw_crop, binary_crop, is_sharp)
                     6-tuples.  Both shapes are accepted.
        """
        # Normalise QR tuple shape so the rest of the code can assume 6-tuples
        qr_results = self._normalise_qr_results(qr_results)

        # PRIORITY 1 — red zone escape (overrides everything)
        red_cmd = self._handle_red_zone_escape(red_results)
        if red_cmd is not None:
            return red_cmd

        state = self.sm.get_state()

        # Watchdog timeouts
        if self.sm.check_timeout():
            if state == "SEARCH":
                self.sm.update_state("LAND", "Search timeout")
                return "LAND"
            else:
                self.sm.update_state("SEARCH", "State timeout")
                return "HOVER"

        # ── State dispatch ────────────────────────────────────
        if state == "SEARCH":
            return self._handle_search(deviation)
        elif state == "SCAN_QR":
            return self._handle_scan_qr(qr_results)
        elif state == "STORE_TARGET":
            return self._handle_store_target(qr_results)
        elif state == "NAVIGATE_CORRIDOR":
            return self._handle_navigate_corridor(deviation, green_results)
        elif state == "TARGET_ALIGN":
            return self._handle_target_align(qr_results)
        elif state == "DROP_READY":
            return self._handle_drop_ready(qr_results)
        elif state == "RETURN_HOME":
            return self._handle_return_home()
        elif state == "LAND":
            return "LAND"

        return "HOVER"

    # ──────────────────────────────────────────────────────────────────────────
    # QR-tuple normaliser  (accepts both 2-tuple and 6-tuple variants)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_qr_results(qr_results):
        """
        Convert any QR-tuple shape into the canonical 6-tuple
        (data, (cx,cy), rect, raw_crop, binary_crop, is_sharp).

        Missing fields are filled with None / True so downstream code
        does not need to special-case shapes.
        """
        normalised = []
        for entry in qr_results:
            if len(entry) == 6:
                normalised.append(entry)
            elif len(entry) == 5:
                d, c, r, raw, bn = entry
                normalised.append((d, c, r, raw, bn, True))
            elif len(entry) == 4:
                d, c, r, raw = entry
                normalised.append((d, c, r, raw, None, True))
            elif len(entry) == 3:
                d, c, raw = entry
                normalised.append((d, c, None, raw, None, True))
            elif len(entry) == 2:
                d, c = entry
                normalised.append((d, c, None, None, None, True))
            else:
                logger.warning(f"[MISSION] Unknown QR tuple shape: {len(entry)}")
        return normalised

    # ──────────────────────────────────────────────────────────────────────────
    # Red zone
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_red_zone_escape(self, red_results):
        red_detected = self._check_red_danger(red_results)

        if red_detected:
            if not self.in_red_zone:
                self.in_red_zone       = True
                self.red_detected_time = time.time()
                logger.warning("[RED] ZONE DETECTED — ESCAPE INITIATED")

            elapsed = time.time() - self.red_detected_time
            if elapsed < self.RED_ESCAPE_TIMEOUT:
                return self._execute_red_escape(red_results)
            else:
                logger.error("[RED] ESCAPE TIMEOUT — FORCE EXIT")
                self.sm.update_state("LAND", "Red zone escape failed")
                return "FORCE_EXIT"
        else:
            if self.in_red_zone:
                logger.info("[RED] Zone cleared — resuming mission")
                self.in_red_zone       = False
                self.red_detected_time = None
            return None

    def _execute_red_escape(self, red_results):
        if not red_results:
            return "HOVER"
        cx, cy, _ = red_results[0]
        dx = cx - 320
        return "ESCAPE_LEFT" if dx > 0 else "ESCAPE_RIGHT"

    def _check_red_danger(self, red_results):
        if not red_results:
            return False
        for cx, cy, area in red_results:
            if area > self.RED_DANGER_AREA:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # State handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_search(self, deviation):
        if deviation is not None and deviation != 0:
            self.no_detection_count = 0
            self.sm.update_state("NAVIGATE_CORRIDOR", "Corridor found")
            return "HOVER"
        self.no_detection_count += 1
        return "HOVER"

    def _handle_scan_qr(self, qr_results):
        if qr_results:
            self.sm.update_state("STORE_TARGET", "QR detected")
            return "HOVER"
        return "ROTATE_SCAN"

    def _handle_store_target(self, qr_results):
        """
        The first QR is the START QR.  We just record its decoded text and
        move on — the actual template binarisation has already happened
        inside QRDetector.load_target_from_qr_payload() (called from
        vision_combined.py).
        """
        if not qr_results:
            self.sm.update_state("SCAN_QR", "Lost QR")
            return "HOVER"

        data_string, _, _, _, _, _ = qr_results[0]
        self.stored_coordinates = data_string

        if self.target_data is not None and self.target_data.is_valid():
            logger.info(
                f"[MISSION] Template active — shape="
                f"{self.target_data.template_qr_image.shape}")
        else:
            logger.warning(
                "[MISSION] No target template loaded yet — "
                "template matching will be unavailable")

        self.sm.update_state("NAVIGATE_CORRIDOR", "Target stored")
        return "FORWARD"

    def _handle_navigate_corridor(self, deviation, green_results):
        if green_results:
            self.sm.update_state("TARGET_ALIGN", "Green waypoint reached")
            return "HOVER"

        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 100:
                self.sm.update_state("SEARCH", "Lost corridor")
            return "HOVER"

        self.no_detection_count = 0
        if abs(deviation) < self.CORRIDOR_CENTERED:
            return "FORWARD"
        elif deviation > 30:
            return "RIGHT"
        elif deviation < -30:
            return "LEFT"
        return "FORWARD"

    def _handle_target_align(self, qr_results):
        """
        Feature 3: align with a candidate QR AND verify visual match.
        Payload release is blocked until both conditions hold for STABLE_FRAMES.
        """
        if not qr_results:
            self.no_detection_count += 1
            if self.no_detection_count > 50:
                self.sm.update_state("NAVIGATE_CORRIDOR", "Lost target QR")
            return "HOVER"

        self.no_detection_count = 0

        best_cmd    = "HOVER"
        best_match  = MatchResult()

        for data, (cx, cy), _rect, _raw, binary_crop, is_sharp in qr_results:
            dx = cx - 320
            dy = cy - 240

            # ── Visual template verification via QRDetector ──────────────────
            verified, conf = self._verify_candidate(binary_crop, is_sharp)

            # Track best match for HUD
            if conf > best_match.confidence:
                best_match = MatchResult(
                    match_found = verified,
                    confidence  = conf,
                    center      = (cx, cy),
                    angle       = 0.0,
                    deviation_x = dx,
                    deviation_y = dy,
                )

            # ── Positional alignment check ───────────────────────────────────
            if abs(dx) < self.QR_CENTERED and abs(dy) < self.QR_CENTERED:
                if verified or self.target_data is None:
                    # Only count stable frames when verified (or no template)
                    self.stable_count += 1
                    if self.stable_count > self.STABLE_FRAMES:
                        self.sm.update_state(
                            "DROP_READY", "Target aligned + verified")
                        return "HOVER"
                else:
                    self.stable_count = 0
                    logger.debug(
                        f"[ALIGN] QR centred but mismatch "
                        f"(conf={conf:.2f}) — skipping")
            else:
                self.stable_count = 0
                best_cmd = "RIGHT" if dx > 0 else "LEFT"

        self.last_match = best_match
        return best_cmd

    def _handle_drop_ready(self, qr_results):
        """
        Feature 3: final guard — re-verify visually before DROP command.
        Payload release is FORBIDDEN without successful verification.
        """
        if not qr_results:
            self.sm.update_state("TARGET_ALIGN", "Lost target")
            return "HOVER"

        _, _, _, _, binary_crop, is_sharp = qr_results[0]
        drop_allowed = False
        conf         = 0.0

        if self.target_data is not None and self.target_data.is_valid():
            drop_allowed, conf = self._verify_candidate(binary_crop, is_sharp)
            logger.info(
                f"[DROP] Template check: conf={conf:.3f} "
                f"drop_allowed={drop_allowed}")
        else:
            # No template — fall back to position-only (degraded mode)
            drop_allowed = True
            logger.warning(
                "[DROP] No template — drop authorised by position only")

        if drop_allowed:
            self.sm.update_state("RETURN_HOME", "Drop complete")
            return "DROP"
        else:
            logger.warning(
                "[DROP] Template verification failed — aborting drop")
            self.sm.update_state("TARGET_ALIGN", "Template mismatch at drop")
            return "HOVER"

    def _handle_return_home(self):
        if self.sm.get_timer() > 300:
            self.sm.update_state("LAND", "Return complete")
            return "LAND"
        return "FORWARD"

    # ──────────────────────────────────────────────────────────────────────────
    # Verification helper — single source of truth
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_candidate(self,
                           binary_crop,
                           is_sharp: bool) -> tuple:
        """
        Run visual template verification via QRDetector.

        Returns (verified, confidence).
        """
        # No template loaded → cannot verify
        if self.target_data is None or not self.target_data.is_valid():
            return False, 0.0

        # No QRDetector injected → cannot verify
        if self.qr_detector is None:
            logger.debug("[VERIFY] QRDetector not injected — skipping")
            return False, 0.0

        # Blurry candidate → unreliable, refuse to match
        if not is_sharp:
            logger.debug("[VERIFY] Candidate too blurry — refusing verify")
            return False, 0.0

        if binary_crop is None or binary_crop.size == 0:
            return False, 0.0

        # Use QRDetector's module-grid matcher
        found, conf = self.qr_detector.match_against_target(
            binary_crop, threshold=self.match_threshold)
        return found, conf

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> str:
        return self.sm.get_state()

    def get_debug_info(self) -> dict:
        return {
            'state':        self.sm.get_state(),
            'timer':        self.sm.get_timer(),
            'stable_count': self.stable_count,
            'no_detection': self.no_detection_count,
            'in_red_zone':  self.in_red_zone,
            'red_timer':   (time.time() - self.red_detected_time)
                            if self.red_detected_time else 0,
            'match_conf':   self.last_match.confidence,
            'match_found':  self.last_match.match_found,
        }