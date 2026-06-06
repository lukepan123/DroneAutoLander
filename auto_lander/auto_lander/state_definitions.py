from enum import IntEnum

class LP_State(IntEnum):
    """State vector for the landing platform."""
    PX  = 0
    PY  = 1
    PZ  = 2
    V   = 3
    A   = 4
    YAW = 5
    YAW_RATE = 6


class LP_Measurement(IntEnum):
    """Measurement vector for the landing platform."""
    PX  = 0
    PY  = 1
    PZ  = 2
    YAW = 3