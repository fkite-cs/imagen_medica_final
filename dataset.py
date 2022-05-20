import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

from torch.utils.data import Dataset

# "dataset/chest_xray/train"
# "dataset/chest_xray/val"

# "dataset/chest_xray/test"

class OurDataset(Dataset):
    def __init__(self, data_path) -> None:
        """
            normal: 0
            pneumonia: 1
        """
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = []
        for i, dir in enumerate(sorted(os.listdir(data_path))):
            for name in glob.glob(os.path.join(data_path, dir, '*')):
                self.data.append((name, i)) # 0 normal, 1 neumonía
            

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx): 
        img_path = self.data[idx][0]
        # leer una etiqueta
        label = self.data[idx][1]
        # leer una imagen
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        img = self.transform(img)
        img = torch.from_numpy(np.array(img))
    
        return img, label 