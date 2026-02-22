import os
import urllib.request
import zipfile
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_URL = "https://storage.googleapis.com/aiolympiadmy/maio2026_qualis/maio2026_sketch_classification/sketch_clf_dataset.zip"

def download_and_extract_data(data_dir="sketch_clf"):
    if not os.path.exists(data_dir):
        print("Downloading dataset ...")
        urllib.request.urlretrieve(DATASET_URL, "sketch_clf_dataset.zip")
        with zipfile.ZipFile("sketch_clf_dataset.zip") as zf:
            zf.extractall(".")
        print("Dataset extracted successfully.")
    else:
        print("Dataset already present.")

class SketchDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir (string): Directory with all the images and CSVs.
            split (string): "train", "valid", or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        if split == "train":
            csv_path = os.path.join(data_dir, "train_labels.csv")
            self.data_frame = pd.read_csv(csv_path)
        elif split == "valid":
            csv_path = os.path.join(data_dir, "validation_labels.csv")
            self.data_frame = pd.read_csv(csv_path)
        elif split == "test":
            # For test set, we just list the images in the test folder in alphabetical order
            test_dir = os.path.join(data_dir, "test")
            self.image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
        else:
            raise ValueError("Split must be 'train', 'valid', or 'test'.")

        # Load class mappings
        classes_path = os.path.join(data_dir, "classes.txt")
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f]
        self.label_to_id = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        if self.split in ["train", "valid"]:
            return len(self.data_frame)
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        if self.split in ["train", "valid"]:
            img_name = self.data_frame.iloc[idx, 0] # Assuming first col is filename
            label_str = self.data_frame.iloc[idx, 1] # Assuming second col is label
            
            # The directory is named 'validation' instead of 'valid'
            dir_name = "validation" if self.split == "valid" else "train"
            img_path = os.path.join(self.data_dir, dir_name, img_name)
            label = self.label_to_id[label_str]
        else:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_dir, "test", img_name)
            label = -1 # Dummy label for test set
            
        # Convert to grayscale 'L' mode and resize to 28x28 just in case, though they should be 28x28
        image = Image.open(img_path).convert('L').resize((28,28))

        if self.transform:
            image = self.transform(image)

        if self.split in ["train", "valid"]:
            return image, label
        else:
            return image, img_name

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_test_transform

if __name__ == "__main__":
    download_and_extract_data()
