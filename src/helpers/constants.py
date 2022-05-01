"""File that stores the constants accessible via the output and input modules. """

# Colors in RGB for OpenCV Text annotations
WHITE: tuple = (255, 255, 255)
BLACK: tuple = (0, 0, 0)
RED: tuple = (255, 0, 0)
BLUE: tuple = (0, 0, 255)
GREEN: tuple = (0, 255, 0)

# Locations for OpenCV Text annotation
LEFT_CORNER1: tuple = (50, 50)
LEFT_CORNER2: tuple = (50, 100)
LEFT_CORNER3: tuple = (50, 150)
LEFT_CORNER4: tuple = (50, 200)
LEFT_CORNER5: tuple = (50, 250)

# Paths to directories
FSVPATH: str = "C:/Users/welleron/Desktop/mmp/datasets/fsv/videos/"
LABDIR: str = "output/labels/csv/"
FVSDIR: str = "output/ml/fvs/"
MLDIR: str = "output/ml/models/"
GRAPHDIR: str = "output/graphs/"
POSEDIR: str = "output/pose/"

# Best Performing models based on the evaluation results per each type of an event
FALL_CLF: str = "output/ml/models/falls_nb.pkl"
SPIN_CLF: str = "output/ml/models/spins_svc.pkl"
JUMP_CLF: str = "output/ml/models/jumps_svc.pkl"
