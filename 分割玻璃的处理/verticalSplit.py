"""
      额外的分割方法尝试：
      1. 先进行垂直线分割
      2. 进行水平线分割
      垂直线分割需要避免相邻边框的影响 → 目前来看阈值为180的效果比较好
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def verticalSplit(file_path):
    # 避免距离较近的结构胶影响分割效果
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

    def find_vertical_lines(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=5)

        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # 垂直线的斜率判断，阈值可以调整
                    vertical_lines.append((x1 + x2) // 2)

        # 返回所有检测到的垂直线的X坐标，返回前进行了排序的操作
        return sorted(filter_close_lines(vertical_lines, 180))

    def crop_images_by_lines(image, line_positions):
        cropped_images = []
        start_x = 0
        for x in line_positions:
            cropped_images.append(image[:, start_x:x])
            start_x = x
        # 添加最后一个窗格
        cropped_images.append(image[:, start_x:])
        return cropped_images

    image = cv2.imread(file_path)

    # 找到垂直线
    vertical_lines = find_vertical_lines(image)

    # 裁剪图像
    cropped_images = crop_images_by_lines(image, vertical_lines)

    # 显示裁剪后的图像
    for idx, img in enumerate(cropped_images):
        plt.subplot(1, len(cropped_images), idx + 1)
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
    verticalSplit(file_path)





