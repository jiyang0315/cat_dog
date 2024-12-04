import os
import random

def generate_train_test_labels(image_folder, train_file, test_file, train_ratio=0.8):
    # 创建或覆盖输出的 txt 文件
    with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
        # 遍历cat和dog文件夹
        for label, folder_name in enumerate(['cat', 'dog']):
            folder_path = os.path.join(image_folder, folder_name)
            
            # 检查文件夹是否存在
            if not os.path.isdir(folder_path):
                print(f"文件夹 {folder_path} 不存在！")
                continue

            # 获取文件夹中所有图像文件
            image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

            # 随机打乱文件列表
            random.shuffle(image_files)

            # 计算训练集和测试集的分割点
            train_size = int(len(image_files) * train_ratio)

            # 写入训练集和测试集
            for i, filename in enumerate(image_files):
                if i < train_size:
                    train_f.write(f"{filename} {label}\n")
                else:
                    test_f.write(f"{filename} {label}\n")

# 调用函数，指定图片文件夹和输出的txt文件路径
image_folder = '/home/jiyang/jiyang/Projects/Cat_Dog/cat_dog'  # 图片所在的主文件夹
train_file = './train_labels.txt'  # 训练集的txt文件路径
test_file = './test_labels.txt'  # 测试集的txt文件路径

generate_train_test_labels(image_folder, train_file, test_file)
print("训练集和测试集标签文件生成成功！")
