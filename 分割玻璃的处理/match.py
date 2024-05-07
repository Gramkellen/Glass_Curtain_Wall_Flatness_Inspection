"""
！！！未完成！！！
该脚本用于读取玻璃幕墙图像，判断相邻玻璃反射图像边缘坐标范围是否匹配。
"""

import cv2
from split import split_glass_wall
from crop import crop_green_edges
from edge import detect_reflected_edges


def match_two_edge(all_edges, adjacents, positions, idx, direction, tolerance=4):
    """
    该函数用于比较两个相邻玻璃的反射边缘是否一致。

    参数:
    - all_edges: 所有玻璃的边缘反射图像坐标。
    - adjacents: 当前玻璃的邻接关系。
    - positions: 所有玻璃的绝对位置。
    - idx:       当前玻璃的边缘反射图像坐标。
    - direction: 当前玻璃需要检测的边缘方向。
    - tolerance: 允许的误差范围，默认值为4

    返回值:
    - 反射边缘一致返回 True，不一致返回 False，没有邻接玻璃返回 None
    """

    # 计算反方西
    if direction == 'up':
        opposite_direction = 'down'
    elif direction == 'down':
        opposite_direction = 'up'
    elif direction == 'left':
        opposite_direction = 'right'
    elif direction == 'right':
        opposite_direction = 'left'
    else:
        return

    # 检查两个边缘是否有邻接
    for adjacent in adjacents[direction]:
        # 只检测坐标大于当前图像的，避免重复检测
        if adjacent > idx:
            # 打印当前窗户和邻接窗户边缘坐标（测试）
            print(all_edges[idx])
            print(all_edges[adjacent])

            # 当前图片反射图像存在direction边缘
            if len(all_edges[idx][direction]):
                # 邻接图片反射图像存在opposite_direction边缘
                if len(all_edges[adjacent][opposite_direction]):
                    # 比较两个邻接窗户的反射边缘相对横坐标
                    if direction == 'up' or direction == 'down':
                        # 现在只考虑边缘只有一组范围的情况，如果测试中发现可能有多段范围，再进行修改！！！
                        cur_edge_l, cur_edge_r = all_edges[idx][direction][0]
                        adj_edge_l, adj_edge_r = all_edges[adjacent][opposite_direction][0]
                        # 两个邻接窗户的横坐标
                        cur_cropped_x, _ = positions[idx]
                        adj_cropped_x, _ = positions[adjacent]
                        # 两个邻接窗户的反射边缘绝对横坐标
                        cur_l = cur_edge_l + cur_cropped_x
                        cur_r = cur_edge_r + cur_cropped_x
                        adj_l = adj_edge_l + adj_cropped_x
                        adj_r = adj_edge_r + adj_cropped_x
                        # 比较横坐标范围是否一致
                        if abs(cur_l - adj_l) < tolerance and abs(cur_r - adj_r) < tolerance:
                            return True
                        else:
                            return False

                    # 比较两个邻接窗户的反射边缘相对纵坐标
                    else:
                        # 现在只考虑边缘只有一组范围的情况，如果测试中发现可能有多段范围，再进行修改！！！
                        cur_edge_u, cur_edge_d = all_edges[idx][direction][0]
                        adj_edge_u, adj_edge_d = all_edges[adjacent][opposite_direction][0]
                        # 两个邻接窗户的纵坐标
                        _, cur_cropped_y = positions[idx]
                        _, adj_cropped_y = positions[adjacent]
                        # 两个邻接窗户的反射边缘绝对横坐标
                        cur_u = cur_edge_u + cur_cropped_y
                        cur_d = cur_edge_d + cur_cropped_y
                        adj_u = adj_edge_u + adj_cropped_y
                        adj_d = adj_edge_d + adj_cropped_y
                        # 比较横坐标范围是否一致
                        if abs(cur_u - adj_u) < tolerance and abs(cur_d - adj_d) < tolerance:
                            return True
                        else:
                            return False

                # 邻接图片反射图像不存在opposite_direction边缘
                else:
                    cur_edge_f, cur_edge_b = all_edges[idx][direction][0]
                    # 如果当前窗户边缘坐标范围小于误差允许
                    if abs(cur_edge_f - cur_edge_b) < tolerance:
                        return True
                    else:
                        return False

            # 当前图片反射图像不存在direction边缘
            else:
                # 邻接图片反射图像存在opposite_direction边缘
                if len(all_edges[adjacent][opposite_direction]):
                    adj_edge_f, adj_edge_b = all_edges[adjacent][opposite_direction][0]
                    # 如果邻接窗户边缘坐标范围小于误差允许
                    if abs(adj_edge_f - adj_edge_b) < tolerance:
                        return True
                    else:
                        return False

                # 邻接图片反射图像不存在opposite_direction边缘
                else:
                    return True
        return


def match_reflected_edges(image):
    # 获取分割后的玻璃图像并得到邻接关系字典
    cropped_images, cropped_positions, adjacency_dict = split_glass_wall(image)

    # 各玻璃的位置信息
    positions = []

    # 存储每个分割后图像的反射图像边缘坐标范围信息
    all_edges = {}

    for idx, img in enumerate(cropped_images):
        # 切除绿色边框
        cropped_img, relative_position = crop_green_edges(img)

        # 计算位置信息
        x, y = cropped_positions[idx]
        relative_x, relative_y = relative_position
        positions.append((x + relative_x, y + relative_y))

        # 获取反射图像边缘信息
        edges, _ = detect_reflected_edges(cropped_img)

        # 存储edges信息
        all_edges[idx] = edges

    # 打印位置信息（测试）
    print("所有玻璃位置信息（左上角坐标）：", positions)

    # 比较相邻图像的反射图像边缘坐标范围是否一致
    for idx in range(len(adjacency_dict)):
        # 得到该图像的邻接关系
        adjacents = adjacency_dict[idx]

        # 检查上侧是否有邻接
        result = match_two_edge(all_edges, adjacents, positions, idx, 'up')
        if result is True:
            print("第", idx, "号和第", adjacents['up'][0], "号玻璃反射边缘一致")
        elif result is False:
            print("第", idx, "号和第", adjacents['up'][0], "号玻璃反射边缘不一致")

        # 检查下侧是否有邻接
        result = match_two_edge(all_edges, adjacents, positions, idx, 'down')
        if result is True:
            print("第", idx, "号和第", adjacents['down'][0], "号玻璃反射边缘一致")
        elif result is False:
            print("第", idx, "号和第", adjacents['down'][0], "号玻璃反射边缘不一致")

        # 检查左侧是否有邻接
        result = match_two_edge(all_edges, adjacents, positions, idx, 'left')
        if result is True:
            print("第", idx, "号和第", adjacents['left'][0], "号玻璃反射边缘一致")
        elif result is False:
            print("第", idx, "号和第", adjacents['left'][0], "号玻璃反射边缘不一致")

        # 检查右侧是否有邻接
        result = match_two_edge(all_edges, adjacents, positions, idx, 'right')
        if result is True:
            print("第", idx, "号和第", adjacents['right'][0], "号玻璃反射边缘一致")
        elif result is False:
            print("第", idx, "号和第", adjacents['right'][0], "号玻璃反射边缘不一致")


if __name__ == "__main__":
    file_path = "data/split1.png"
    image = cv2.imread(file_path)
    match_reflected_edges(image)
