'''
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2,os
 
model = YOLO('/home/lwf/nanting/yolov8/ultralytics-main/runs/segment/dataset30+yolov8x_d3/weights/best.pt')
results = model('/home/lwf/nanting/yolov8/ultralytics-main/murals/degradation3/3_164_8-256.jpeg')
image = Image.open("/home/lwf/nanting/yolov8/ultralytics-main/murals/degradation3/3_164_8-256.jpeg")

for result in results:
    boxes = result.boxes          # 输出的检测框
    masks = result.masks          # 输出的掩码信息
 
def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
            inside = not inside
        j = i
    return inside
 
def find_polygon_pixels(masks_xy, boxes_cls):    # 所有掩码像素点及其对应类别属性的列表
    # 初始化存储所有像素点和类别属性的列表
    all_pixels_with_cls = []
 
    # 遍历每个多边形
    for i, polygon in enumerate(masks_xy):
        cls = boxes_cls[i]  # 当前多边形的类别属性
 
        # 将浮点数坐标点转换为整数类型
        polygon = [(int(point[0]), int(point[1])) for point in polygon]
 
        # 找出当前多边形的边界框
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_y = min(point[1] for point in polygon)
        max_y = max(point[1] for point in polygon)
 
        # 在边界框内遍历所有像素点
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # 检查像素点是否在多边形内部
                if is_point_inside_polygon(x, y, polygon):
                    # 将像素点坐标和类别属性组合成元组，添加到列表中
                    all_pixels_with_cls.append(((x, y), cls))
 
    return all_pixels_with_cls
 
def reconstruct_image(image_size, pixels_with_cls):
    # 创建一个和图片原始大小相同的黑色图像
    reconstructed_image = np.ones((image_size[1], image_size[0]), dtype=np.uint8)
    for pixel, cls in pixels_with_cls:
        if cls >= 0:
            reconstructed_image[pixel[1], pixel[0]] = 0
        else:
            reconstructed_image[pixel[1], pixel[0]] = 1  
    return reconstructed_image
 
 
masks_xy = masks.xy    # 每个掩码的边缘点坐标
boxes_cls = boxes.cls  # 每个多边形的类别属性
 
# 调用函数找出每个多边形内部的点和相应的类别属性
all_pixels_with_cls = find_polygon_pixels(masks_xy, boxes_cls)
image_size = image.size
 
#print("所有像素点和相应的类别属性：", all_pixels_with_cls)  # 在终端显示所有掩码对应的坐标以及对应的属性元组

reconstructed_image = reconstruct_image(image_size, all_pixels_with_cls)   # 重建图像
#Image.fromarray(reconstructed_image).save("mask/3_164_8-2.png")   # 保存图像 
np.save('npy/3_164_8-256.npy', reconstructed_image)
# Show the results
for r in results:
    im_array = r.plot()                        # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.save('mask/3-142-13.png')                     # save image
'''

#from msilib import sequence
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
 
def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
            inside = not inside
        j = i
    return inside
 
def find_polygon_pixels(masks_xy, boxes_cls):    # 所有掩码像素点及其对应类别属性的列表
    # 初始化存储所有像素点和类别属性的列表
    all_pixels_with_cls = []
 
    # 遍历每个多边形
    for i, polygon in enumerate(masks_xy):
        cls = boxes_cls[i]  # 当前多边形的类别属性

        # 将浮点数坐标点转换为整数类型
        polygon = [(int(point[0]), int(point[1])) for point in polygon]
        #print(polygon)
        # 找出当前多边形的边界框
        min_x = min(point[0] for point in polygon)#1#
        max_x = max(point[0] for point in polygon)#512#
        min_y = min(point[1] for point in polygon)#1#
        max_y = max(point[1] for point in polygon)#512#
 
        # 在边界框内遍历所有像素点
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # 检查像素点是否在多边形内部
                if is_point_inside_polygon(x, y, polygon):
                    # 将像素点坐标和类别属性组合成元组，添加到列表中
                    all_pixels_with_cls.append(((x, y), cls))
 
    return all_pixels_with_cls
 
def reconstruct_image(image_size, pixels_with_cls):
    #创建一个和图片大小相同的数组
    reconstructed_image = np.ones((image_size[1], image_size[0]), dtype=np.uint8)
    for pixel, cls in pixels_with_cls:
        if cls >= 0:
            reconstructed_image[pixel[1], pixel[0]] = 0
        else:
            reconstructed_image[pixel[1], pixel[0]] = 1  
    return reconstructed_image
 
def file_exists(filename):
    return os.path.exists(filename)

# 获取RGB图像的路径
image_dir = "ultralytics-main/murals/134_1"
 
# 遍历每个图片文件
for image_filename in os.listdir(image_dir):

    sketch_path='ultralytics-main/npy/npy_134'

    reconstructe_image_filename = f"{image_filename.split('_png')[0]}.npy"  # 重建mask文件名
    reconstructe_image_path = os.path.join(sketch_path, reconstructe_image_filename)  # 重建mask保存路径
    if file_exists(reconstructe_image_path):
        continue
    else:

        if image_filename.endswith('.png'):
            image_path = os.path.join(image_dir, image_filename)
        
        # 执行模型预测
            model = YOLO('ultralytics-main/runs/segment/dataset30+yolov8x_d1/weights/best.pt')
            results = model(image_path)
            image = Image.open(image_path)
 
        # 提取掩码和检测框信息
            for result in results:
                boxes = result.boxes          # 输出的检测框
                masks = result.masks          # 输出的掩码信息
 
            masks_xy = masks.xy    # 每个掩码的边缘点坐标
            boxes_cls = boxes.cls  # 每个多边形的类别属性
 
        # 调用函数找出每个mask内部的点和相应的类别属性
            all_pixels_with_cls = find_polygon_pixels(masks_xy, boxes_cls)
 
        # 对每一张图像的分割掩码进行重建并保存在特定的文件夹中
            image_size = image.size
            reconstructed_image = reconstruct_image(image_size, all_pixels_with_cls)  # mask
            reconstructed_image_filename = f"{image_filename.split('_rgb.png')[0]}.npy" # 重建mask文件名
            reconstructed_image_path = os.path.join(sketch_path, reconstructed_image_filename)  # 重建mask保存路径
            np.save(reconstructed_image_path, reconstructed_image)
            #Image.fromarray(reconstructed_image).save(reconstructed_image_path)  # 保存图像

'''
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
 
def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1])):
            inside = not inside
        j = i
    return inside
 
def find_polygon_pixels(masks_xy, boxes_cls):    # 所有掩码像素点及其对应类别属性的列表
    # 初始化存储所有像素点和类别属性的列表
    all_pixels_with_cls = []
 
    # 遍历每个多边形
    for i, polygon in enumerate(masks_xy):
        cls = boxes_cls[i]  # 当前多边形的类别属性
 
        # 将浮点数坐标点转换为整数类型
        polygon = [(int(point[0]), int(point[1])) for point in polygon]
 
        # 找出当前多边形的边界框
        min_x = min(point[0] for point in polygon)
        max_x = max(point[0] for point in polygon)
        min_y = min(point[1] for point in polygon)
        max_y = max(point[1] for point in polygon)
 
        # 在边界框内遍历所有像素点
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # 检查像素点是否在多边形内部
                if is_point_inside_polygon(x, y, polygon):
                    # 将像素点坐标和类别属性组合成元组，添加到列表中
                    all_pixels_with_cls.append(((x, y), cls))
 
    return all_pixels_with_cls
 
def reconstruct_image(image_size, pixels_with_cls):
    # 创建一个和图片原始大小相同的黑色图像
    reconstructed_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
 
    # 将属性为 0 的像素点设为绿色，属性为 1 的像素点设为蓝色 ，其余的像素点默认为背景设为黑色
    for pixel, cls in pixels_with_cls:
        if cls == 0:
            reconstructed_image[pixel[1], pixel[0]] = [0, 255, 0]  # 绿色
        elif cls == 1:
            reconstructed_image[pixel[1], pixel[0]] = [0, 0, 255]  # 蓝色
        else:
            reconstructed_image[pixel[1], pixel[0]] = [0, 0, 0]    # 黑色
 
    return reconstructed_image
 
# 获取RGB图像的路径
image_dir = "murals/degradation3_256"
 
# 遍历每个图片文件
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_dir, image_filename)
        
        # 执行模型预测
        model = YOLO('/home/lwf/nanting/yolov8/ultralytics-main/runs/segment/dataset30+yolov8x_d3/weights/best.pt')
        results = model(image_path)
        image = Image.open(image_path)
 
        # 提取掩码和检测框信息
        for result in results:
            boxes = result.boxes          # 输出的检测框
            masks = result.masks          # 输出的掩码信息
 
        masks_xy = masks.xy    # 每个掩码的边缘点坐标
        boxes_cls = boxes.cls  # 每个多边形的类别属性
 
        # 调用函数找出每个mask内部的点和相应的类别属性
        all_pixels_with_cls = find_polygon_pixels(masks_xy, boxes_cls)
 
        # 对每一张图像的分割掩码进行重建并保存在特定的文件夹中
        image_size = image.size
       
        reconstructed_image = reconstruct_image(image_size, all_pixels_with_cls)  # 重建图像
        reconstructed_image_filename = f"{image_filename.split('_rgb.png')[0]}_mask.png"  # 重建图像文件名
        reconstructed_image_path = os.path.join('/home/lwf/nanting/yolov8/ultralytics-main/mask/mask256', reconstructed_image_filename)  # 重建图像保存路径
        Image.fromarray(reconstructed_image).save(reconstructed_image_path)  # 保存图像
'''