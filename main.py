import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_data, preprocess_data, load_answers, create_dataloader
from model import get_model
from utils import visualize_results, save_prediction_as_tif

# 定义数据路径
data_dir = 'F:/AI/notebookcode/CCF/data/trainset/CCF大数据与计算智能大赛数据集'
answer_dir = 'F:/AI/notebookcode/CCF/data/answer'
submit_example_dir = 'F:/AI/notebookcode/CCF/data/submit_example'

# 加载数据
dsm, dom, green, nir, red, rededge, meta = load_data()
answers = load_answers()

# 数据预处理
multi_band_image_normalized = preprocess_data(dom, green, nir, red, rededge)

# 创建数据集和数据加载器
images = np.expand_dims(multi_band_image_normalized, axis=0)  # 假设我们只有一个图像
labels = answers
dataloader = create_dataloader(images, labels)

# 定义模型
model = get_model(num_classes=3, pretrained=True)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# 评估模型
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 打印分类报告
print(classification_report(all_labels, all_predictions))

# 可视化结果
predicted_image = np.array(all_predictions).reshape(dom.shape[1], dom.shape[2])
visualize_results(dom[0], predicted_image)

# 保存预测结果
output_path = 'submission/results/predicted_results.tif'
save_prediction_as_tif(predicted_image, output_path, meta)

# 保存模型
model_path = 'submission/model/corn_disaster_model.pth'
torch.save(model.state_dict(), model_path)

print('Model training and prediction completed successfully.')
print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")

print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")

print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")

print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")

print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")

print("Importing modules...")
print("Modules imported successfully.")
print("Starting main program...")
print("Main program completed.")
