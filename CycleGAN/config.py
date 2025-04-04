import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:/Users/pc/Downloads/horse_zebra/horse2zebra/horse2zebra"
VAL_DIR = "C:/Users/pc/Downloads/horse_zebra/horse2zebra/horse2zebra"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "logs/gen_H/"
CHECKPOINT_GEN_Z = "logs/gen_Z/"
CHECKPOINT_DISC_H = "logs/disc_H/"
CHECKPOINT_DISC_Z = "logs/disc_Z/"

# CHECKPOINT_GEN_H = "genh.pth.tar"
# CHECKPOINT_GEN_Z = "genz.pth.tar"
# CHECKPOINT_CRITIC_H = "critich.pth.tar"
# CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)