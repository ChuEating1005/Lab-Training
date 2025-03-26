import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Define transforms
test_tfm = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_tfm = v2.Compose([
    v2.RandomResizedCrop(128, scale=(0.9, 1)),
    v2.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
    v2.RandomGrayscale(p=0.25),
    v2.RandomRotation(15),           # 隨機旋轉 ±15 度
    v2.ColorJitter(                  # 隨機調整圖像顏色
        brightness=0.2,                      # 亮度
        contrast=0.2,                        # 對比度
        saturation=0.2,                      # 飽和度
        hue=0.03                             # 色調
    ),
    v2.RandomAffine(                 # 隨機仿射變換
        degrees=0,                           # 旋轉角度
        translate=(0.1, 0.1),               # 平移
        scale=(0.9, 1.1)                    # 縮放
    ),
    v2.RandomErasing(p=0.2),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im, label