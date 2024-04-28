from abc import abstractmethod
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
from enum import Enum
from utils import file_operations
import zipfile
from PIL import Image 
import io

class CXRLabel(Enum):
    NORMAL = 0
    ABNORMAL = 1

class CXRDataset(Dataset):
  def __init__(self, archive_file):
    self.archive_file = archive_file
    self.image_to_label = {}
    self.extract_data()
    self.items = list(self.image_to_label.items())

  @abstractmethod
  def extract_data(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    img_path, label = self.items[idx]
    with open(img_path, "rb") as file:
      img = Image.open(io.BytesIO(file.read())).convert('RGB')

      transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
      ])
      
      img = transform(img)
      return img, label.value
    
  def __len__(self):
    return len(self.items)

class COVID19_Radiography(CXRDataset):
  """
  └── COVID-19_Radiography_Dataset/
      ├── COVID/
      │   ├── images/
      │   └── masks/
      ├── COVID.metadata.xlsx
      ├── Lung Opacity/
      │   ├── images/
      │   └── masks/
      ├── Lung_Opacity.metadata.xlsx
      ├── Viral Pneumonia/
      │   ├── images/
      │   └── masks/
      ├── Viral Pneumonia.metadata.xlsx
      ├── Normal/
      │   ├── images/
      │   └── masks/
      ├── Normal.metadata.xlsx
      └── README.md.txt
    """

  def __init__(self):
    super().__init__("datasets/COVID-19_Radiography.zip")

  def extract_data(self):
    self.normal = ["Normal"]
    self.abnormal = ["COVID", "Lung_Opacity", "Viral Pneumonia"]
    images_directories = []

    with zipfile.ZipFile(self.archive_file, "r") as zip_ref:
      zip_ref.extractall()

    for category in (self.normal + self.abnormal):
      metadata_file = os.path.join(os.getcwd(), 'COVID-19_Radiography_Dataset', f'{category}.metadata.xlsx')
      image_dir = os.path.join(os.getcwd(), 'COVID-19_Radiography_Dataset', category, 'images')
      metadata = pd.read_excel(metadata_file)
      
      for index, row in metadata.iterrows():
        dataset_id = f"{category}_{row['URL']}"

        file_path = os.path.join(image_dir, f"{row['FILE NAME']}.png")
        if file_operations.is_valid_image_file(file_path):

          # Read the image file from the zip
          with open(file_path, "rb") as file:
            try:
              img_data = file.read()
              img = Image.open(io.BytesIO(img_data))
              img.verify()

              self.image_to_label[file_path] = CXRLabel.NORMAL if category in self.normal else CXRLabel.ABNORMAL

            except (IOError, SyntaxError) as e:
              logging.info(f"Invalid image file {filename}: {e}, skipping..")
              continue

          


              