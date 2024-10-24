import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the image size and batch size
IMAGE_SIZE = (224, 224)  # GFNet 输入通常是 224x224 的大小
BATCH_SIZE = 32

# Load and preprocess images using tf.data.Dataset
def load_images(directory):
    images = []
    labels = []
    # 遍历目录并存储文件路径
    for label in os.listdir(directory):
        label_folder = os.path.join(directory, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                file_path = os.path.join(label_folder, filename)
                images.append(file_path)  # 存储文件路径
                labels.append(label)      # 存储标签
    return images, labels

# TensorFlow 数据加载器 - 用于预处理图像
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)  # 调整图像大小以匹配 GFNet 输入
    # 对于label的定义
    label = tf.cast(label, tf.int32)
    img = img / 255.0  # 归一化像素值
    # 
    return img, label

# 构建用于 GFNet 的数据集
def build_dataset(image_paths, labels, shuffle=True):
    # 将字符串标签转换为数值索引
    unique_labels = list(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in labels]

    # 从文件路径和标签中创建 TensorFlow 数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, numerical_labels))
    
    # 映射数据集以应用预处理函数
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # 批处理和预取以提高性能
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# 获取测试集
def get_test_dataset():
    test_images, test_labels = load_images('test')
    # test_dataset = build_dataset(test_images, test_labels, shuffle=False)
    test_dataset = build_dataset(test_images, test_labels)
    return test_dataset

# 切分并获取训练和验证集 -- 每个类别的分布要趋近于一致且具有随机性
def get_train_validation_dataset():
    train_images, train_labels = load_images('train')

    # 将字符串标签转换为数值索引
    # 随机性划分，使其不具有偏向性
    unique_labels = list(set(train_labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    numerical_labels = [label_to_index[label] for label in train_labels]

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, numerical_labels, test_size=0.15, random_state=42, stratify=numerical_labels 
    )
    train_dataset = build_dataset(train_images, train_labels)
    val_dataset = build_dataset(val_images, val_labels, shuffle=False)
    return train_dataset, val_dataset

def extract_labels_from_dataset(dataset):
    all_labels = []
    for _, labels in dataset.unbatch():
        all_labels .append(labels.numpy())
    return np.array(all_labels)

# 测试数据加载器的功能
if __name__ == "__main__":
    train_dataset, val_dataset = get_train_validation_dataset()
    test_dataset = get_test_dataset()

    # 显示一批图像
    for images, labels in train_dataset.take(1):  # 仅获取一批图像 (?)
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.show()
