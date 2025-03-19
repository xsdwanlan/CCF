import os
import rasterio
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip

# 定义数据路径
data_dir = 'F:/AI/notebookcode/CCF/data/trainset/CCF大数据与计算智能大赛数据集'
answer_dir = 'F:/AI/notebookcode/CCF/data/answer'
submit_example_dir = 'F:/AI/notebookcode/CCF/data/submit_example'

# 加载所有相关文件
def load_data():
    dsm_path = os.path.join(data_dir, 'dsm.tif')
    result_path = os.path.join(data_dir, 'result.tif')
    green_path = os.path.join(data_dir, 'result_Green.tif')
    nir_path = os.path.join(data_dir, 'result_NIR.tif')
    red_path = os.path.join(data_dir, 'result_Red.tif')
    rededge_path = os.path.join(data_dir, 'result_RedEdge.tif')
    
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        dsm_meta = src.meta
    
    with rasterio.open(result_path) as src:
        dom = src.read()
        dom_meta = src.meta
    
    with rasterio.open(green_path) as src:
        green = src.read(1)
    
    with rasterio.open(nir_path) as src:
        nir = src.read(1)
    
    with rasterio.open(red_path) as src:
        red = src.read(1)
    
    with rasterio.open(rededge_path) as src:
        rededge = src.read(1)
    
    return dsm, dom, green, nir, red, rededge, dom_meta

# 数据预处理
def preprocess_data(dom, green, nir, red, rededge):
    # 合并多波段影像
    multi_band_image = np.stack([dom, green, nir, red, rededge], axis=0)
    
    # 归一化
    multi_band_image_normalized = (multi_band_image - np.min(multi_band_image)) / (np.max(multi_band_image) - np.min(multi_band_image))
    
    return multi_band_image_normalized

# 创建数据集类
class CornDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载答案集
def load_answers():
    answer_path = os.path.join(answer_dir, 'standard.tif')
    with rasterio.open(answer_path) as src:
        answers = src.read(1)
    return answers

# 数据增强
transform = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    RandomVerticalFlip()
])

# 创建数据集和数据加载器
def create_dataloader(images, labels, transform=transform, batch_size=32):
    dataset = CornDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader