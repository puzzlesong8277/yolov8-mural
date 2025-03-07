import os
from PIL import Image

def resize_images(input_folder, output_folder):
    # 创建一个存放修改后图片的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".gif"):
            img = Image.open(os.path.join(input_folder, filename)) # 打开输入文件夹中的图像文件

            new_size = (256, 256)  

            img_resized = img.resize(new_size) # 将图像大小调整为指定的大小
            img_resized.save(os.path.join(output_folder, filename)) # 将调整后的图像保存到输出文件夹中

 
input_folder = '/home/lwf/nanting/yolov8/ultralytics-main/murals/degradation3'
output_folder = '/home/lwf/nanting/yolov8/ultralytics-main/murals/degradation3_256'

# 执行函数resize_images,调整图片大小并保存到输出文件夹中
resize_images(input_folder, output_folder)