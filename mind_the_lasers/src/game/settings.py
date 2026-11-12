WIDTH = 1000
HEIGHT = 600
FPS = 60

BALL_RADIUS = 18
BALL_SPEED = 220 #100 is a good number
BALL_SPEED_TRAINING = 280

MAX_LIVES = 5
RESET_LIVES_EVERY_LEVEL = True
HIT_FLASH_TIME = 1.0
INVINCIBLE_TIME = 0.8

BG_COLOR = (25, 25, 35)
BALL_COLOR = (70, 170, 255)
HIT_COLOR = (255, 70, 70)
LASER_COLOR = (255, 60, 60)
GOAL_COLOR = (100, 255, 100)
TEXT_COLOR = (240, 240, 240)
HEART_COLOR = (230, 50, 70)
EMPTY_HEART_COLOR = (80, 80, 90)

BOOST_PER_LEVEL = 20
MAX_BOOST_LEVEL = 5
BOOST_CHARGE_TIME = 1.0
BOOST_DURATION = 1.0

TRAINING_TRIALS_PER_COMMAND = 5
TRAINING_COMMAND_DELAY = 2.0

#ErrP settings
CALIBRATION_TRIALS = 0
ERRP_TRIALS_PER_COMMAND = 25 # Number of approximate trials per left/right (prob_error = 50%) command in the ErrP session e.g. total trials = ERRP_TRIALS_PER_COMMAND * 2
# Above setting is not intuitive and is an anachronism, fix this.

INCORRECT_DISTANCE_THRESHOLD_FRACTION = 3 # Interactive mode distance to consider the command incorrect: cuts half the screen into fractions and chooses the one closest to the center. The denominator is this value.

ERRP_COMMAND_DELAY = 3.0 # Observation mode delay before the command is given
ERRP_INCORRECT_FEEDBACK_TIME = 1.3 # Time until display incorrect feedback
ERRP_INCORRECT_FEEDBACK_TIME_CONTINUOUS = 0.0 # Time until display incorrect feedback for continuous mode

ERRP_CORRECT_TRIAL_DELAY = 0.3 # Time until next trial after correct command
ERRP_INTER_TRIAL_MIN_TIME = 1.0 # Minimum pause before SPACE can start the next trial (seconds)

ERROR_PROBABILITY = 0.2
LR_PROBABILITY = 0.5

DEBUG_ERRP = False

# Marker transmission
MARKER_TRANSPORT = "udp"  # "lsl" or "udp"

# Unicorn BCI Core Recorder UDP trigger input
MARKER_UDP_IP = "127.0.0.1"
MARKER_UDP_PORT = 1000