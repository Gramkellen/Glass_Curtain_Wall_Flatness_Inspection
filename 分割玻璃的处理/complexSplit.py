"""
      额外的分割方法尝试：
      1. 先进行垂直线分割
      2. 进行水平线分割
      垂直线分割需要避免相邻边框的影响 → 目前来看阈值为180的效果比较好
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_column(matrix, new_col):
    if matrix.size == 0:
        # 如果矩阵是空的，初始化矩阵为第一列
        matrix = np.array(new_col).reshape(-1, 1)
    else:
        # 如果矩阵已经有数据，按列堆叠
        new_col = np.array(new_col).reshape(-1, 1)
        matrix = np.hstack((matrix, new_col))

    return matrix

# 过滤掉接近的边线
def filter_close_lines(lines, min_distance):
    """过滤掉距离过近的垂直线"""
    if not lines:
        return []

    # 首先按位置排序
    sorted_lines = sorted(lines)

    # 过滤过近的线
    filtered_lines = [sorted_lines[0]]  # 添加第一条线
    for line in sorted_lines[1:]:
        if line - filtered_lines[-1] >= min_distance:
            filtered_lines.append(line)
    return filtered_lines

# 按照水平和竖直，利用Canny检测去查找边缘线
def find_lines(image, orientation='vertical', line_length=100, line_gap=5, min_distance = 180):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 根据方向调整霍夫变换的角度范围
    theta = np.pi / 180 if orientation == 'vertical' else np.pi / 2

    lines = cv2.HoughLinesP(edges, 1, theta, 80, minLineLength=line_length, maxLineGap=line_gap)

    line_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == 'vertical' and abs(x2 - x1) < 10:  # 垂直线
                line_positions.append((x1 + x2) // 2)
            elif orientation == 'horizontal' and abs(y2 - y1) < 10:  # 水平线
                line_positions.append((y1 + y2) // 2)

    # 排序并过滤掉过于接近的线
    return sorted(filter_close_lines(line_positions, min_distance))

# 裁剪得到对应的图片，并进行返回
def crop_images_by_orientation(image, line_positions, orientation):
    cropped_images = []
    start = 0

    for pos in line_positions:
        if orientation == 'vertical':
            cropped_images.append(image[:, start:pos])
        elif orientation == 'horizontal':
            cropped_images.append(image[start:pos, :])
        start = pos

    # 添加最后一个区域
    if orientation == 'vertical':
        cropped_images.append(image[:, start:])
    elif orientation == 'horizontal':
        cropped_images.append(image[start:, :])

    return cropped_images

# 多层次的复杂分割函数实现
def complexSplit(file_path):
    image = cv2.imread(file_path)
    # 先进行垂直分割
    vertical_lines = find_lines(image, 'vertical')

    # 裁剪图像
    vertically_cropped_images = crop_images_by_orientation(image, vertical_lines, 'vertical')

    # 移除掉两侧的图像
    if(len(vertically_cropped_images) >= 2):
        vertically_cropped_images.pop(0)
        vertically_cropped_images.pop(-1)
    # 显示裁剪后的图像
    # 对每个垂直分割的部分应用水平分割
    all_cropped_images = []
    matrix = np.empty((0, 0))
    for v_img in vertically_cropped_images:
        row = 0
        horizontal_lines = find_lines(v_img, 'horizontal',min_distance = 100)
        horizontally_cropped_images = crop_images_by_orientation(v_img, horizontal_lines, 'horizontal')
        # 过滤掉面积过小的图像
        new_col = []
        for h_img in horizontally_cropped_images:
            if h_img.size > 100000:  # 面积阈值，根据需要调整
                all_cropped_images.append(h_img)
                new_col.append(row)
                row += 1

        if row != 0 :
            matrix = add_column(matrix, new_col)
    print(matrix)
    # 显示结果
    plt.figure(figsize=(12, 8))
    for idx, img in enumerate(all_cropped_images):
        plt.subplot(1, len(all_cropped_images), idx + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

def detect_grid(image_path, area_threshold):
    image = cv2.imread(image_path)
    # 转化为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # 定义函数计算交点
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # 计算两个向量的行列式
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        det1 = det([x1, y1], [x2, y2])
        det2 = det([x3, y3], [x4, y4])
        div = det([x1 - x2, y1 - y2], [x3 - x4, y3 - y4])

        # 确保除数不为零
        if div == 0:
            return np.inf, np.inf  # 不相交或平行线

        d = (det1, det2)
        x = det(d, [x1 - x2, x3 - x4]) / div
        y = det(d, [y1 - y2, y3 - y4]) / div
        return x, y

    # 计算所有交点
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            p = line_intersection(line1, line2)
            if 0 <= p[0] < gray.shape[1] and 0 <= p[1] < gray.shape[0]:  # 确保交点在图像内
                intersections.append(p)

    # 转换为numpy数组方便计算
    intersections = np.array(intersections)

    # 使用面积阈值过滤交点
    if len(intersections) == 0:
        print("No intersections were detected.")
        return

    # 计算所有点的成对距离
    dist_matrix = np.sqrt(np.sum((intersections[:, np.newaxis] - intersections[np.newaxis, :]) ** 2, axis=2))
    # 筛选距离在一定阈值内的点作为有效交点
    valid_intersections = intersections[np.all(dist_matrix > area_threshold, axis=1)]

    # 绘制直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制交点
    for p in valid_intersections:
        cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)

    # 使用matplotlib显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Grid with Area Threshold')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    file_path = "data/split.png"
    complexSplit(file_path)





