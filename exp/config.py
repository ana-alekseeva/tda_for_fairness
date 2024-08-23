# Model parameters
BASE_MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = 'tomh/toxigen_hatebert'
TOKENIZER_NAME = 'bert-base-uncased'

TEST_SAMPLES_PER_GROUP = 50

# Data parameters
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01

#NUM_CHECKPOINTS_TO_KEEP = 20
#CHECKPOINT_DIR = 'checkpoints'
#CHECKPOINT_PREFIX = 'model_checkpoint_'

# Device
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
#SAVE_PATH = './finetuned_toxigen_hatebert'