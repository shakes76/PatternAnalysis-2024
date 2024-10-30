import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

class GlobalFilterLayer(layers.Layer):
    """全局频率滤波层"""
    def __init__(self, **kwargs):
        super(GlobalFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化可学习的频域滤波器
        self.filters = self.add_weight(
            name='freq_filters',
            shape=input_shape[1:],  # 匹配输入特征图的形状
            initializer='glorot_uniform',
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.01)  # 添加L2正则化
        )

    def call(self, x):
        # 转换到频域
        x_freq = tf.signal.fft2d(tf.cast(x, tf.complex64))
        
        # 应用频域滤波器
        x_filtered = x_freq * tf.cast(self.filters, tf.complex64)
        
        # 转换回空间域
        x_spatial = tf.signal.ifft2d(x_filtered)
        
        return tf.math.real(x_spatial)

def build_model(input_shape=(224,224,3)):
    # 输入层和数据增强
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2)
    ])(inputs)

    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg')

    x = base_model.output

    # 添加GFNet特有的全局频率滤波
    x = GlobalFilterLayer()(x)

    # Gradient usually use "relu"
    x = layers.Dense(128, activation='relu')(x) #128 output dimensions (neurons)
    x = layers.Dropout(0.5)(x)
    # To solve a binary classification problem
    # softmax/sigmod - focus on binary or multi-classification/just 2-classification
    predictions = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model