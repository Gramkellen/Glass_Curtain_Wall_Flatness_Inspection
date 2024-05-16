'''
  这只是一个尝试 —— 可全部修改

  注：分割算法的尝试都可以放到这里
'''



import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 重新加载图像
image_path = "../data/test1.png"
image = Image.open(image_path)
image_np = np.array(image)
hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

# 更精确地定义绿色的HSV范围
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# 使用新的范围创建绿色掩码
green_mask_hsv = cv2.inRange(hsv_image, lower_green, upper_green)

# 使用形态学操作去除噪声并闭合边框
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
green_mask_cleaned = cv2.morphologyEx(green_mask_hsv, cv2.MORPH_CLOSE, kernel)
green_mask_cleaned = cv2.morphologyEx(green_mask_cleaned, cv2.MORPH_OPEN, kernel)

# 再次查找轮廓
contours_cleaned, _ = cv2.findContours(green_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤掉小的轮廓
min_contour_area = 1000  # 最小面积阈值
filtered_contours = [contour for contour in contours_cleaned if cv2.contourArea(contour) > min_contour_area]

# 在原始图像上绘制过滤后的轮廓
filtered_contour_image = image_np.copy()
cv2.drawContours(filtered_contour_image, filtered_contours, -1, (255, 0, 0), 2)

# 提取并保存每个绿色边框内的分割区域
filtered_segments = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    segment = image_np[y:y+h, x:x+w]
    filtered_segments.append(segment)

# 显示清理后的掩码和带有轮廓的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("清理后的绿色掩码")
plt.imshow(green_mask_cleaned, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("带有过滤后轮廓的图像")
plt.imshow(filtered_contour_image)

plt.show()

# 显示过滤后的分割区域
grid_size = int(np.ceil(np.sqrt(len(filtered_segments))))

plt.figure(figsize=(15, 15))
for i, segment in enumerate(filtered_segments):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(segment)
    plt.axis('off')
plt.show()
