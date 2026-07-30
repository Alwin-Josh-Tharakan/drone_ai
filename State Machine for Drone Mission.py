# state_machine.py — updated with RED_ESCAPE state and extended timeouts
import logging

logger = logging.getLogger(__name__)


class StateMachine:
    def __init__(self):
        self.state      = "SEARCH"
        self.prev_state = None
        self.state_timer = 0

        # Timeouts in frames @ 30 fps
        self.TIMEOUT_SEARCH         = 1800   # 60 s
        self.TIMEOUT_ALIGN          = 600    # 20 s
        self.TIMEOUT_SCAN_QR        = 450    # 15 s
        self.TIMEOUT_STORE_TARGET   = 150    # 5 s
        self.TIMEOUT_NAVIGATE       = 1800   # 60 s
        self.TIMEOUT_TARGET_ALIGN   = 450    # 15 s
        self.TIMEOUT_DROP_READY     = 300    # 10 s
        self.TIMEOUT_RED_ESCAPE     = 300    # 10 s (backup; mission logic has its own timer)

    def update_state(self, new_state: str, reason: str = "") -> None:
        if new_state != self.state:
            self.prev_state  = self.state
            self.state       = new_state
            self.state_timer = 0
            logger.info(f"[STATE] {self.prev_state} → {self.state}"
                        + (f" | {reason}" if reason else ""))
        else:
            self.state_timer += 1

    def check_timeout(self) -> bool:
        t = self.state_timer
        s = self.state
        if   s == "SEARCH"           and t > self.TIMEOUT_SEARCH:         return True
        elif s == "ALIGN_CORRIDOR"   and t > self.TIMEOUT_ALIGN:           return True
        elif s == "SCAN_QR"          and t > self.TIMEOUT_SCAN_QR:         return True
        elif s == "STORE_TARGET"     and t > self.TIMEOUT_STORE_TARGET:    return True
        elif s == "NAVIGATE_CORRIDOR"and t > self.TIMEOUT_NAVIGATE:        return True
        elif s == "TARGET_ALIGN"     and t > self.TIMEOUT_TARGET_ALIGN:    return True
        elif s == "DROP_READY"       and t > self.TIMEOUT_DROP_READY:      return True
        elif s == "RED_ESCAPE"       and t > self.TIMEOUT_RED_ESCAPE:      return True
        return False

    def get_state(self) -> str:
        return self.state

    def get_timer(self) -> int:
        return self.state_timer