"""Mission Logic Controller"""

from state_machine import StateMachine

class MissionController:
    def __init__(self, target_qr="TARGET_A"):
        self.sm = StateMachine()
        self.target_qr = target_qr
        
        # Thresholds
        self.RED_DANGER_AREA = 5000
        self.CORRIDOR_CENTERED = 20
        self.GREEN_CENTERED = 30
        self.QR_CENTERED = 15
        self.STABLE_FRAMES = 90  # 3s at 30fps
        
        # Counters
        self.stable_count = 0
        self.no_detection_count = 0
        
    def process(self, qr_results, green_results, red_results, deviation):
        """Main decision logic - returns command string"""
        
        # PRIORITY 1: RED ZONE (override everything)
        if self._check_red_danger(red_results):
            self.sm.update_state("SEARCH", "RED ZONE DETECTED")
            return self._handle_red_avoid(red_results)
        
        # Get current state
        state = self.sm.get_state()
        
        # Check timeout
        if self.sm.check_timeout():
            if state == "SEARCH":
                self.sm.update_state("LAND", "Search timeout")
                return "LAND"
            else:
                self.sm.update_state("SEARCH", "State timeout")
                return "HOVER"
        
        # State handlers
        if state == "SEARCH":
            return self._handle_search(deviation)
        
        elif state == "ALIGN_CORRIDOR":
            return self._handle_align_corridor(deviation, green_results)
        
        elif state == "FOLLOW_CORRIDOR":
            return self._handle_follow_corridor(deviation, green_results)
        
        elif state == "DETECT_TARGET":
            return self._handle_detect_target(qr_results, green_results)
        
        elif state == "VERIFY_TARGET":
            return self._handle_verify_target(qr_results)
        
        elif state == "DROP_PAYLOAD":
            return self._handle_drop_payload(qr_results)
        
        elif state == "RETURN_HOME":
            return self._handle_return_home()
        
        elif state == "LAND":
            return "LAND"
        
        return "HOVER"
    
    # ========== STATE HANDLERS ==========
    
    def _handle_search(self, deviation):
        """Search for corridor"""
        if deviation is not None and deviation != 0:
            self.no_detection_count = 0
            self.sm.update_state("ALIGN_CORRIDOR", "Corridor found")
            return "HOVER"
        
        self.no_detection_count += 1
        return "HOVER"
    
    def _handle_align_corridor(self, deviation, green_results):
        """Align with corridor center"""
        # Check for green (skip to target detection)
        if green_results and len(green_results) > 0:
            self.sm.update_state("DETECT_TARGET", "Green banner detected")
            return "HOVER"
        
        # No corridor
        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 100:
                self.sm.update_state("SEARCH", "Lost corridor")
            return "HOVER"
        
        self.no_detection_count = 0
        
        # Check if centered
        if abs(deviation) < self.CORRIDOR_CENTERED:
            self.sm.update_state("FOLLOW_CORRIDOR", "Corridor aligned")
            return "FORWARD"
        
        # Align
        if deviation > 0:
            return "RIGHT"
        else:
            return "LEFT"
    
    def _handle_follow_corridor(self, deviation, green_results):
        """Follow corridor and look for green banner"""
        # Check for green
        if green_results and len(green_results) > 0:
            self.sm.update_state("DETECT_TARGET", "Green banner detected")
            return "HOVER"
        
        # No corridor
        if deviation is None:
            self.no_detection_count += 1
            if self.no_detection_count > 100:
                self.sm.update_state("SEARCH", "Lost corridor")
            return "HOVER"
        
        self.no_detection_count = 0
        
        # Navigate
        if abs(deviation) < self.CORRIDOR_CENTERED:
            return "FORWARD"
        elif deviation > 30:
            return "RIGHT"
        elif deviation < -30:
            return "LEFT"
        else:
            return "FORWARD"
    
    def _handle_detect_target(self, qr_results, green_results):
        """Scan for QR code near green banner"""
        if qr_results and len(qr_results) > 0:
            self.sm.update_state("VERIFY_TARGET", "QR detected")
            return "HOVER"
        
        # Center on green banner while scanning
        if green_results and len(green_results) > 0:
            green_center, area = green_results[0]
            cx, cy = green_center
            frame_center = (320, 240)
            
            dx = cx - frame_center[0]
            dy = cy - frame_center[1]
            
            if abs(dx) > self.GREEN_CENTERED:
                return "RIGHT" if dx > 0 else "LEFT"
            else:
                return "HOVER"
        
        return "HOVER"
    
    def _handle_verify_target(self, qr_results):
        """Verify QR matches target"""
        if not qr_results or len(qr_results) == 0:
            self.no_detection_count += 1
            if self.no_detection_count > 50:
                self.sm.update_state("DETECT_TARGET", "Lost QR")
            return "HOVER"
        
        self.no_detection_count = 0
        qr_data, qr_center = qr_results[0]
        
        # Check if correct target
        if qr_data == self.target_qr:
            self.sm.update_state("DROP_PAYLOAD", f"Target verified: {qr_data}")
            self.stable_count = 0
            return "HOVER"
        else:
            self.sm.update_state("FOLLOW_CORRIDOR", f"Wrong QR: {qr_data}")
            return "FORWARD"
    
    def _handle_drop_payload(self, qr_results):
        """Align precisely and drop"""
        if not qr_results or len(qr_results) == 0:
            self.no_detection_count += 1
            if self.no_detection_count > 50:
                self.sm.update_state("DETECT_TARGET", "Lost QR during drop")
            return "HOVER"
        
        self.no_detection_count = 0
        qr_data, qr_center = qr_results[0]
        cx, cy = qr_center
        frame_center = (320, 240)
        
        dx = cx - frame_center[0]
        dy = cy - frame_center[1]
        
        # Check if centered and stable
        if abs(dx) < self.QR_CENTERED and abs(dy) < self.QR_CENTERED:
            self.stable_count += 1
            if self.stable_count > self.STABLE_FRAMES:
                self.sm.update_state("RETURN_HOME", "Drop complete")
                return "DROP"
            else:
                return "HOVER"
        else:
            self.stable_count = 0
            if abs(dx) > self.QR_CENTERED:
                return "RIGHT" if dx > 0 else "LEFT"
            else:
                return "HOVER"
    
    def _handle_return_home(self):
        """Return to start position"""
        if self.sm.get_timer() > 300:  # 10s return
            self.sm.update_state("LAND", "Return complete")
            return "LAND"
        return "FORWARD"
    
    # ========== HELPERS ==========
    
    def _check_red_danger(self, red_results):
        """Check if red zone is dangerous"""
        if not red_results or len(red_results) == 0:
            return False
        
        for red_center, area in red_results:
            if area > self.RED_DANGER_AREA:
                return True
        return False
    
    def _handle_red_avoid(self, red_results):
        """Avoid red zone"""
        if not red_results or len(red_results) == 0:
            return "HOVER"
        
        # Find closest red zone
        red_center, area = red_results[0]
        cx, cy = red_center
        frame_center = (320, 240)
        
        # Move away from red
        if cx < frame_center[0]:
            return "RIGHT"
        else:
            return "LEFT"
    
    def get_state(self):
        """Return current state"""
        return self.sm.get_state()
    
    def get_debug_info(self):
        """Return debug information"""
        return {
            'state': self.sm.get_state(),
            'timer': self.sm.get_timer(),
            'stable_count': self.stable_count,
            'no_detection': self.no_detection_count
        }