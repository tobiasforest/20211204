# A reproduction using PyTorch on the paper:
# UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS
# https://hal-enpc.archives-ouvertes.fr/hal-01864755

import torch


BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.001
IMAGE_SIZE = 224
DEVICE = 'cuda'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = './models/'
BETA1 = 0.5
TOP_K = 5
