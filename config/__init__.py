"""Game configuration — re-exports all constants for convenient access."""

# Game states
SETUP, BATTLE, GAME_OVER = 0, 1, 2

# Layout
WINDOW_WIDTH = 1060
WINDOW_HEIGHT = 680
GRID_COLS = 11
GRID_ROWS = 9
HEX_SIZE = 32
GRID_OFFSET_X = 210
GRID_OFFSET_Y = 80

from .colors import *
from .units import UNIT_TYPES, UNIT_TAGS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES
from .presets import PRESETS
from .timing import *
