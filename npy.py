'''
#将npy文件转换为图片
import PIL
import numpy as np
from PIL import Image
path = "ultralytics-main/npy/npy_1007/mask/3_147_7.png.npy"

loaded = np.load(path)
print(loaded.shape)
image = Image.fromarray(loaded)
image.show()


#下载npy文件
#导入所需的包
import numpy as np  

#导入npy文件路径位置
test = np.load('ultralytics-main/npy/npy_0918/1_119_1.squ.txt.npy')
np.savetxt('ultralytics-main/npy/1_119_1.squ.txt', test, delimiter=" ")
print(test.shape)

print(test)

import numpy as np
import matplotlib.pyplot as plt
depthmap = np.load('ultralytics-main/npy/npy_1007/mask/3_147_7.png.npy') #使用numpy载入npy文件
plt.imshow(depthmap) #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
plt.colorbar() #添加colorbar
plt.savefig('depthmap.jpg') #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
plt.show() #在线显示图像

#若要将图像存为灰度图，可以执行如下两行代码
import scipy.misc
scipy.misc.imsave("depth.png", depthmap)




import numpy 
from PIL import Image
import os
 
def pretreatment(image):
    ima=Image.open(image)
    ima=ima.convert('L')         #转化为灰度图像
    im=numpy.array(ima)        #转化为二维数组
    for i in range(im.shape[0]):#转化为二值矩阵
        for j in range(im.shape[1]):
            if im[i,j]<=50:
                im[i,j]=1
            else:
                im[i,j]=0
    return im

# 获取RGB图像的路径
image_dir = "ultralytics-main/murals/test_ref"
 
# 遍历每个图片文件
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_dir, image_filename)
        im=pretreatment(image_path)  #调用图像预处理函数
        sketch_image_filename = f"{image_filename.split('_rgb.sketch')[0]}.npy"  # 重建mask文件名
        sketch_image_path = os.path.join('ultralytics-main/npy/npy_1007/sketch', sketch_image_filename)  # 重建mask保存路径
        numpy.save(sketch_image_path,im)
''' 
import numpy as np
import os
from PIL import Image

# 加载.npy文件
npy_data = np.load('ultralytics-main/npy/npy_3_80/3_1_30.npy') # 替换为你的npy文件路径

# 确保数据是 0 和 1 组成的
assert np.all((npy_data == 0) | (npy_data == 1)), "数据必须是由0和1组成的"
npy_data = np.where(npy_data ==1,0,1)
# 将 numpy 数组转换为图片
# 可以使用 'L' 模式（灰度图像）来存储 0 和 1 对应的黑白图像
image = Image.fromarray(npy_data.astype(np.uint8) * 255)

# 保存图像
image.save('3_1_30.png')

# 显示图像
image.show()
'''
import os
import numpy as np
from PIL import Image

def npy_to_image(npy_file, output_dir):
    # 加载npy文件
    data = np.load(npy_file)

    # 检查数据类型是否为0或1
    if not np.all(np.isin(data, [0, 1])):
        raise ValueError(f"文件 {npy_file} 中的值不是仅由0和1组成")

    # 将数据转换为图像：0为黑色，1为白色
    # 0 -> 0 (黑色), 1 -> 255 (白色)
    img = Image.fromarray((data.astype(np.uint8) * 255)) # 0->0黑色，1->255白色
    
    #npy_data = np.where(data ==1,0,1)
# 将 numpy 数组转换为图片
# 可以使用 'L' 模式（灰度图像）来存储 0 和 1 对应的黑白图像
    #img = Image.fromarray(npy_data.astype(np.uint8) * 255)
    # 获取文件名并修改扩展名为.png
    img_name = os.path.basename(npy_file).replace('.png.npy', '_mask.png')

    # 保存图片到目标目录
    img.save(os.path.join(output_dir, img_name))
    print(f"文件 {npy_file} 转换并保存为 {img_name}")

def convert_npy_files(input_dir, output_dir):
# 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录下的所有npy文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            npy_file_path = os.path.join(input_dir, file_name)
            npy_to_image(npy_file_path, output_dir)

# 示例：转换指定文件夹下的所有npy文件
input_directory = 'ultralytics-main/npy/npy_1007/mask' # 输入npy文件所在目录
output_directory = 'ultralytics-main/mask/mask1225' # 输出图片文件夹目录
convert_npy_files(input_directory, output_directory)
'''