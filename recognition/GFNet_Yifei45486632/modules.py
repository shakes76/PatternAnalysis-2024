import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

class GlobalFilterLayer(layers.Layer):
   def __init__(self, **kwargs):
       super(GlobalFilterLayer, self).__init__(**kwargs)
       
   def build(self, input_shape):
       # 初始化可学习的频域滤波器
       self.filter_shape = input_shape[1:]
       self.filters = self.add_weight(
           name='frequency_filters',
           shape=self.filter_shape,
           initializer='random_normal',
           trainable=True
       )
       
   def call(self, x):
       # 转换到频域
       x_freq = tf.signal.fft2d(tf.cast(x, tf.complex64))
       
       # 应用可学习滤波器
       x_filtered = x_freq * tf.cast(self.filters, tf.complex64)
       
       # 转换回空间域
       x_spatial = tf.signal.ifft2d(x_filtered)
       
       return tf.math.real(x_spatial)

def gfnet_block(x, filters):
    residual = x
    
    # 添加维度调整层
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
    
    # 标准化
    x = layers.LayerNormalization()(x)
    
    # 全局滤波
    x = GlobalFilterLayer()(x)
    
    # FFN
    x = layers.Conv2D(filters * 4, 1, activation='gelu')(x)
    x = layers.Conv2D(filters, 1)(x)
    
    # 添加残差连接（现在维度匹配）
    return layers.Add()([x, residual])

def build_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    # 数据增强
    x = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2)
    ])(inputs)
    
    # 基础特征提取
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base_model(x)
    
    # GFNet blocks with matching dimensions
    x = layers.Conv2D(256, 1)(x)  # 初始维度调整
    x = gfnet_block(x, 256)
    x = gfnet_block(x, 512)
    
    # 后续层保持不变
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)