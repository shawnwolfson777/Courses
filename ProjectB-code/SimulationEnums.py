from enum import Enum

class Delay(Enum):
    IDEAL = 0 # Communication without delay and erasures
    GAUSSIAN = 1 # Communication with gaussian distributed delay