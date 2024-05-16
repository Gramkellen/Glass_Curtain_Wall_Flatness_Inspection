"""
该脚本用于分割出图像中每一扇窗户。只保留完整的窗户。
"""

import cv2


def split_glass_wall(image):
    """
    该函数用于将玻璃幕墙图像分割为一扇扇窗户。

    参数:
    - image: 玻璃幕墙图像（已完成反射分割和边框检测）。

    返回值:
    - cropped_images: 裁剪后的窗户图像列表。
    - cropped_positions: 裁剪后的窗户位置信息 (x, y) 列表。
    - adjacency_dict: 图片的邻接关系字典。
    """

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值化处理 (green has medium intensity on grayscale)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # 树检索方法去查询轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours]

    # 定义了一个函数来过滤掉面积小于指定最小值的轮廓。
    def filter_contours_by_area(contours, min_area):
        return [c for c in contours if cv2.contourArea(c) > min_area]

    # 找到面积最大的轮廓并删除它
    max_contour = max(contours, key=cv2.contourArea)
    contours = [c for c in contours if c is not max_contour]

    # 使用删除后的最大面积作为新的面积标准
    max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
    min_area = 0.03 * max_area  # 90% of the max contour area

    filtered_contours = filter_contours_by_area(contours, min_area)

    # 存储裁剪窗户的位置信息(x, y)
    cropped_positions = []

    def crop_image_by_contour(image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_positions.append((x, y))
        return image[y:y + h, x:x + w]

    cropped_images = [crop_image_by_contour(image, c) for c in filtered_contours]

    # --------------------相邻性判断-------------------------------------
    # 判断box2相对于box1的方向
    def are_adjacent(box1, box2, tolerance=50):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # box1和box2在同一行
        if abs(y1 - y2) <= tolerance:
            # box2在box1右侧
            if abs(x1 + w1 - x2) <= tolerance:
                return 'right'
            # box2在box1左侧
            elif abs(x2 + w2 - x1) <= tolerance:  # box1 is on the right of box2
                return 'left'
        # box1和box2在同一列
        elif abs(x1 - x2) <= tolerance:
            # box2在box1下侧
            if abs(y1 + h1 - y2) <= tolerance:
                return 'down'
            # box2在box1上侧
            elif abs(y2 + h2 - y1) <= tolerance:
                return 'up'
        return None

    # 得到相反方向
    def invert_direction(direction):
        if direction == 'left':
            return 'right'
        elif direction == 'right':
            return 'left'
        elif direction == 'up':
            return 'down'
        elif direction == 'down':
            return 'up'
        return None

    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

    # 利用字典存储相邻的信息及其位置关系
    adjacency_dict = {i: {'left': [], 'right': [], 'up': [], 'down': []} for i in range(len(bounding_boxes))}

    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            direction = are_adjacent(bounding_boxes[i], bounding_boxes[j])
            if direction:
                adjacency_dict[i][direction].append(j)
                adjacency_dict[j][invert_direction(direction)].append(i)

    return cropped_images, cropped_positions, adjacency_dict


if __name__ == "__main__":
    file_path = "data/test1.png"
    image = cv2.imread(file_path)
    cropped_images, cropped_positions, adjacency_dict = split_glass_wall(image)

    # 显示得到的图片
    for idx, img in enumerate(cropped_images):
        cv2.namedWindow('splited image', cv2.WINDOW_NORMAL)
        cv2.imshow("splited image", img)
        cv2.waitKey(0)

    # 打印位置信息列表
    print("位置信息列表：")
    print(cropped_positions)

    # 打印邻接字典
    print("邻接字典判断相邻性：")
    for i, adjacents in adjacency_dict.items():
        print(f"{i}: {adjacents}")

    # 使用示例
    print("\n使用示例：")
    # 读取第1张图片，并显示它的邻接图片
    idx = 0
    adjacents = adjacency_dict[idx]
    print(f"图像 {idx} 的邻接图片索引：{adjacents}")

    # 显示第1张图片
    # cv2.namedWindow('Window Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Window Image', cropped_images[idx])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

