"""State Machine for Drone Mission"""

class StateMachine:
    def __init__(self):
        self.state = "SEARCH"
        self.prev_state = None
        self.state_timer = 0
        
        # State timeouts (frames at 30fps)
        self.TIMEOUT_SEARCH = 1800        # 60s
        self.TIMEOUT_ALIGN = 600          # 20s
        self.TIMEOUT_DETECT_TARGET = 450  # 15s
        self.TIMEOUT_VERIFY = 150         # 5s
        self.TIMEOUT_DROP = 300           # 10s
        
    def update_state(self, new_state, reason=""):
        """Change state and reset timer"""
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.state_timer = 0
            if reason:
                print(f"[STATE] {self.prev_state} → {self.state} | {reason}")
        else:
            self.state_timer += 1
    
    def check_timeout(self):
        """Check if current state has timed out"""
        if self.state == "SEARCH" and self.state_timer > self.TIMEOUT_SEARCH:
            return True
        elif self.state == "ALIGN_CORRIDOR" and self.state_timer > self.TIMEOUT_ALIGN:
            return True
        elif self.state == "DETECT_TARGET" and self.state_timer > self.TIMEOUT_DETECT_TARGET:
            return True
        elif self.state == "VERIFY_TARGET" and self.state_timer > self.TIMEOUT_VERIFY:
            return True
        elif self.state == "DROP_PAYLOAD" and self.state_timer > self.TIMEOUT_DROP:
            return True
        return False
    
    def get_state(self):
        """Return current state"""
        return self.state
    
    def get_timer(self):
        """Return current state timer"""
        return self.state_timer