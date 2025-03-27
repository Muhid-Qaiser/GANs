import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, sketch_dir, face_dir):
        self.sketch_dir = sketch_dir
        self.face_dir = face_dir

        self.sketch_files = sorted(os.listdir(self.sketch_dir))
        self.face_files = sorted(os.listdir(self.face_dir))

        # Ensure the number of sketches matches the number of face images
        assert len(self.sketch_files) == len(self.face_files), "Number of sketch and face images must match!"


    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):

        # Get file names for sketch and face images
        sketch_file = self.sketch_files[index]
        face_file = self.face_files[index]

         # Full paths to the images
        sketch_path = os.path.join(self.sketch_dir, sketch_file)
        face_path = os.path.join(self.face_dir, face_file)

        # Load the images and convert to numpy arrays
        sketch_image = np.array(Image.open(sketch_path))
        face_image = np.array(Image.open(face_path))

        # Apply augmentations to both images
        augmentations = config.both_transform(image=sketch_image, image0=face_image)
        sketch_image, face_image = augmentations["image"], augmentations["image0"]

        # Additional transformations
        sketch_image = config.transform_only_input(image=sketch_image)["image"]
        face_image = config.transform_only_mask(image=face_image)["image"]

        return sketch_image, face_image  # Return the input-target pair
    

if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
