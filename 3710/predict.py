import numpy as np

# 加载 .npz 文件
npz_file = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
data = np.load(npz_file)

# 打印文件中包含的数据项名称
print("Data keys:", data.files)
labels = data['target']
print("Labels shape:", labels.shape)
print("Labels sample:", labels[:10])  # 打印前10个标签以查看格式