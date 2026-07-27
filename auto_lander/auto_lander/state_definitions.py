from enum import IntEnum


class QUAD_State(IntEnum):
    """State vector for the quadcopter"""

    X = 0
    Y = 1
    Z = 2
    YAW = 3
    

class LP_State(IntEnum):
    """State vector for the landing platform."""

    PX = 0
    PY = 1
    PZ = 2
    V = 3
    A = 4
    YAW = 5
    YAW_RATE = 6


class LP_Measurement(IntEnum):
    """Measurement vector for the landing platform."""

    PX = 0
    PY = 1
    PZ = 2
    YAW = 3
