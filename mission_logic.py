# mission_logic.py
class MissionController:
    __slots__ = ('state', 'prev_state', 'target_qr', 'state_timer', 'stable_count', 
                 'no_detection_count', 'RED_DANGER_AREA', 'CORRIDOR_CENTERED', 
                 'GREEN_CENTERED', 'QR_CENTERED', 'STABLE_FRAMES')
    
    def __init__(self, target_qr="DROP_ZONE_A", match_threshold=None):
        # match_threshold kept for API compatibility with vision_combined.py;
        # this controller decides purely on QR payload + detection geometry.
        self.state = "TAKEOFF"  # Start with takeoff
        self.prev_state = None
        self.target_qr = target_qr
        
        # Counters
        self.state_timer = 0
        self.stable_count = 0
        self.no_detection_count = 0
        
        # Thresholds
        self.RED_DANGER_AREA = 5000
        self.CORRIDOR_CENTERED = 20
        self.GREEN_CENTERED = 30
        self.QR_CENTERED = 15
        self.STABLE_FRAMES = 90
    
    def process(self, qr_results, green_results, red_results, deviation):
        self.state_timer += 1
        
        # PRIORITY 1: RED ZONE OVERRIDE
        if self._check_red_danger(red_results):
            if self.state != "AVOID_RED":
                self.prev_state = self.state
                self.state = "AVOID_RED"
            return self._handle_red_avoid(red_results)
        
        if self.state == "AVOID_RED":
            self.state = self.prev_state or "SEARCH_GREEN"
        
        # State handlers matching mission flow
        if self.state == "TAKEOFF":
            return self._handle_takeoff()
        elif self.state == "SCAN_START_QR":
            return self._handle_scan_start_qr(qr_results)
        elif self.state == "SEARCH_GREEN":
            return self._handle_search_green(green_results)
        elif self.state == "ALIGN_GREEN":
            return self._handle_align_green(green_results)
        elif self.state == "ENTER_CORRIDOR":
            return self._handle_enter_corridor(deviation)
        elif self.state == "FOLLOW_CORRIDOR":
            return self._handle_follow_corridor(deviation, green_results)
        elif self.state == "EXIT_CORRIDOR":
            return self._handle_exit_corridor()
        elif self.state == "SEARCH_TARGET_QR":
            return self._handle_search_target_qr(qr_results, red_results)
        elif self.state == "VERIFY_TARGET":
            return self._handle_verify_target(qr_results)
        elif self.state == "ALIGN_DROP":
            return self._handle_align_drop(qr_results)
        elif self.state == "DROP_PAYLOAD":
            return self._handle_drop_payload(qr_results)
        elif self.state == "RETURN_SEARCH_GREEN":
            return self._handle_return_search_green(green_results)
        elif self.state == "RETURN_CORRIDOR":
            return self._handle_return_corridor(deviation)
        elif self.state == "RTL":
            return self._handle_rtl()
        elif self.state == "LAND":
            return self._handle_land()
        elif self.state == "COMPLETE":
            return "HOVER"
        
        return "HOVER"
    
    # Mission Step 1: Autonomous Takeoff
    def _handle_takeoff(self):
        if self.state_timer > 150:  # 5s takeoff
            self._transition("SCAN_START_QR")
            return "HOVER"
        return "TAKEOFF"
    
    # Mission Step 2: Initial QR Code Scan (at 5m altitude)
    def _handle_scan_start_qr(self, qr_results):
        if qr_results:
            # Decode delivery location from QR
            self.target_qr = qr_results[0][0]  # Set target from scanned QR
            self._transition("SEARCH_GREEN")
            return "FORWARD_1M"  # Move forward 1m after scan
        
        if self.state_timer > 300:  # 10s timeout
            self._transition("SEARCH_GREEN")  # Proceed anyway
        return "HOVER"
    
    # Mission Step 3: Corridor Entry Detection (Green Banner)
    def _handle_search_green(self, green_results):
        if green_results:
            self._transition("ALIGN_GREEN")
            return "HOVER"
        
        if self.state_timer > 600:  # 20s timeout
            self._transition("ENTER_CORRIDOR")  # Proceed blind
        return "SEARCH_PATTERN"
    
    def _handle_align_green(self, green_results):
        if not green_results:
            self.no_detection_count += 1
            if self.no_detection_count > 100:
                self._transition("SEARCH_GREEN")
            return "HOVER"
        
        self.no_detection_count = 0
        green_center, area = green_results[0]
        cx, cy = green_center
        dx = cx - 320
        dy = cy - 240
        
        if abs(dx) < self.GREEN_CENTERED and abs(dy) < self.GREEN_CENTERED:
            self._transition("ENTER_CORRIDOR")
            return "DESCEND_3M"  # Descend to 3m for corridor
        
        return "ALIGN_CENTER"
    
    # Mission Step 4: Autonomous Corridor Navigation (3m altitude)
    def _handle_enter_corridor(self, deviation):
        if deviation is not None and abs(deviation) < self.CORRIDOR_CENTERED:
            self._transition("FOLLOW_CORRIDOR")
            return "FORWARD"
        
        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 100:
                self._transition("SEARCH_GREEN")
            return "HOVER"
        
        return "RIGHT" if deviation > 0 else "LEFT"
    
    def _handle_follow_corridor(self, deviation, green_results):
        # Check if exiting corridor (no more corridor detected for extended period)
        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 150:  # Exited corridor
                self._transition("EXIT_CORRIDOR")
                return "ASCEND_10M"  # Ascend to 10m at delivery zone
            return "HOVER"
        
        self.no_detection_count = 0
        
        if abs(deviation) < self.CORRIDOR_CENTERED:
            return "FORWARD"
        elif deviation > 30:
            return "RIGHT"
        elif deviation < -30:
            return "LEFT"
        return "FORWARD"
    
    # Mission Step 5: Target Identification & Restricted Zone Avoidance (10m altitude)
    def _handle_exit_corridor(self):
        if self.state_timer > 90:  # 3s to ascend
            self._transition("SEARCH_TARGET_QR")
        return "HOVER"
    
    def _handle_search_target_qr(self, qr_results, red_results):
        if qr_results:
            self._transition("VERIFY_TARGET")
            return "HOVER"
        
        if self.state_timer > 450:  # 15s timeout
            self._transition("RTL")  # Abort mission
        
        return "SEARCH_PATTERN"
    
    def _handle_verify_target(self, qr_results):
        if not qr_results:
            self.no_detection_count += 1
            if self.no_detection_count > 50:
                self._transition("SEARCH_TARGET_QR")
            return "HOVER"
        
        self.no_detection_count = 0
        qr_data, qr_center = qr_results[0]
        
        if qr_data == self.target_qr:
            self._transition("ALIGN_DROP")
            self.stable_count = 0
            return "DESCEND_5M"  # Descend to 5m for drop
        else:
            # Wrong QR, continue searching
            self._transition("SEARCH_TARGET_QR")
            return "FORWARD"
    
    # Mission Step 6: Precision Payload Delivery (5m altitude)
    def _handle_align_drop(self, qr_results):
        if not qr_results:
            self.no_detection_count += 1
            if self.no_detection_count > 50:
                self._transition("SEARCH_TARGET_QR")
            return "HOVER"
        
        self.no_detection_count = 0
        qr_data, qr_center = qr_results[0]
        cx, cy = qr_center
        dx = cx - 320
        dy = cy - 240
        
        if abs(dx) < self.QR_CENTERED and abs(dy) < self.QR_CENTERED:
            self.stable_count += 1
            if self.stable_count > self.STABLE_FRAMES:
                self._transition("DROP_PAYLOAD")
                return "LOWER_PAYLOAD"
            return "HOVER"
        else:
            self.stable_count = 0
            if abs(dx) > self.QR_CENTERED:
                return "RIGHT" if dx > 0 else "LEFT"
            return "HOVER"
    
    def _handle_drop_payload(self, qr_results):
        if self.state_timer > 180:  # 6s payload release
            self._transition("RETURN_SEARCH_GREEN")
            return "ASCEND_10M"  # Ascend to 10m for return
        return "RELEASE_PAYLOAD"
    
    # Mission Step 7: Corridor Entry Detection (Return Lap)
    def _handle_return_search_green(self, green_results):
        if green_results:
            self._transition("RETURN_CORRIDOR")
            return "DESCEND_3M"  # Descend to 3m for return corridor
        
        if self.state_timer > 600:  # 20s timeout
            self._transition("RTL")  # Direct RTL
        return "SEARCH_PATTERN"
    
    # Mission Step 8: Autonomous Return Flight
    def _handle_return_corridor(self, deviation):
        # Check if exited corridor (reached start point)
        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 150:
                self._transition("RTL")
                return "HOVER"
            return "HOVER"
        
        self.no_detection_count = 0
        
        if abs(deviation) < self.CORRIDOR_CENTERED:
            return "FORWARD"
        elif deviation > 30:
            return "RIGHT"
        elif deviation < -30:
            return "LEFT"
        return "FORWARD"
    
    # Mission Step 9: Safe Autonomous Landing
    def _handle_rtl(self):
        if self.state_timer > 300:  # 10s return flight
            self._transition("LAND")
        return "RTL"
    
    def _handle_land(self):
        if self.state_timer > 180:  # 6s landing
            self._transition("COMPLETE")
        return "LAND"
    
    def _check_red_danger(self, red_results):
        if not red_results:
            return False
        return any(area > self.RED_DANGER_AREA for _, area in red_results)
    
    def _handle_red_avoid(self, red_results):
        if not red_results:
            return "HOVER"
        red_center, area = red_results[0]
        cx = red_center[0]
        return "RIGHT" if cx < 320 else "LEFT"
    
    def _transition(self, new_state):
        self.prev_state = self.state
        self.state = new_state
        self.state_timer = 0
    
    def get_state(self):
        return self.state
    
    def get_debug_info(self):
        return {
            'state': self.state,
            'timer': self.state_timer,
            'stable_count': self.stable_count,
            'no_detection': self.no_detection_count,
            'target_qr': self.target_qr
        }