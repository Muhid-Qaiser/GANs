import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:/Users/pc/Downloads/sketch_to_face/train"
VAL_DIR = "C:/Users/pc/Downloads/sketch_to_face/val"
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 2000
LOAD_MODEL = False
SAVE_MODEL = True
# CHECKPOINT_DISC = "disc.pth.tar"
# CHECKPOINT_GEN = "gen.pth.tar"

CHECKPOINT_DISC = "logs/discriminator/"
CHECKPOINT_GEN = "logs/generator/"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
