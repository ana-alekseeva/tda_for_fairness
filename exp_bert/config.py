# Model parameters
BASE_MODEL_NAME = 'bert-base-uncased'
TOKENIZER_NAME = 'bert-base-uncased'

# Data parameters
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training parameters
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01