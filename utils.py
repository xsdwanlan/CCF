import matplotlib.pyplot as plt
import numpy as np
import rasterio

# 可视化结果
def visualize_results(original_image, predicted_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_image, cmap='gray')
    plt.title('Predicted Results')
    plt.show()

# 保存预测结果为 TIF 文件
def save_prediction_as_tif(predicted_image, output_path, meta):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=predicted_image.shape[0],
        width=predicted_image.shape[1],
        count=1,
        dtype=predicted_image.dtype,
        crs=meta['crs'],
        transform=meta['transform']
    ) as dst:
        dst.write(predicted_image, 1)