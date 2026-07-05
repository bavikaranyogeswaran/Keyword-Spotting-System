# The 8 voice commands the model will learn to recognize
TARGET_WORDS = [
    "yes",
    "no",
    "stop",
    "go",
    "left",
    "right",
    "up",
    "down",
]

# Audio is recorded at 16,000 samples per second (standard for speech)
SAMPLE_RATE = 16000

# Each clip is standardized to exactly 1 second (16,000 samples)
# CNNs need a fixed input size, so every clip must be the same length
NUM_SAMPLES = 16000

# Number of Mel frequency bins — higher means more frequency detail
N_MELS = 64

# Number of audio clips processed together in one training step
BATCH_SIZE = 64

# Number of times the model sees the full training dataset
EPOCHS = 10

# How fast the model updates its weights during training
LEARNING_RATE = 0.001

# Where the trained model weights are saved
MODEL_PATH = "ml/models/keyword_cnn.pth"

# Where the dataset is downloaded and stored
DATA_ROOT = "ml/data"
