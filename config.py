# =========================
# PATHS
# =========================
DATA_PATH = "data/raw/HS"
CSV_PATH = "data/raw/HS.csv"

PROCESSED_PATH = "data/processed/"
SPECTROGRAM_PATH = "data/processed/spectrograms/"

MODEL_PATH = "models/"
PLOT_PATH = "outputs/plots/"

# =========================
# AUDIO SETTINGS
# =========================
SAMPLE_RATE = 22050     # standard audio sampling rate
DURATION = 3            # seconds (fixed length for all audio)
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128            # spectrogram height

# =========================
# MODEL SETTINGS
# =========================
IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 30

# =========================
# TRAIN SETTINGS
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42