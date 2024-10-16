'''
Various networks in Haiku
'''
import haiku as hk
import jax

#conv layers
class NormConv1D(hk.Module):
    '''
    Batch Normed Conv 1D
    Supports dilation via rate that results in a dilated convolution
    '''
    def __init__(self, channels, stride, rate, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.conv_0 = hk.Conv1D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        self.bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

    def __call__(self, x, is_training, test_local_stats=False):
        out = self.conv_0(x)
        out = self.bn_0(out, is_training, test_local_stats)
        out = jax.nn.relu(out)

        return out

class NormConv1DTranspose(hk.Module):
    '''
    Batch Normed Conv 1D Transpose
    '''
    def __init__(self, channels, stride, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.conv_0 = hk.Conv1DTranspose(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        self.bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

    def __call__(self, x, is_training, test_local_stats=False):
        out = self.conv_0(x)
        out = self.bn_0(out, is_training, test_local_stats)
        out = jax.nn.relu(out)

        return out

class NormConv2D(hk.Module):
    '''
    Batch Normed Conv 2D
    Supports dilation via rate that results in a dilated convolution
    '''
    def __init__(self, channels, stride, rate, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        self.bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

    def __call__(self, x, is_training, test_local_stats=False):
        out = self.conv_0(x)
        out = self.bn_0(out, is_training, test_local_stats)
        out = jax.nn.relu(out)

        return out
        
class NormConv2DTranspose(hk.Module):
    '''
    Batch Normed Conv 2D Transpose
    '''
    def __init__(self, channels, stride, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.conv_0 = hk.Conv2DTranspose(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        self.bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

    def __call__(self, x, is_training, test_local_stats=False):
        out = self.conv_0(x)
        out = self.bn_0(out, is_training, test_local_stats)
        out = jax.nn.relu(out)

        return out

#conv blocks
class NormBlock(hk.Module):
    '''
    Normed Conv Block
    Supports dilation via rate that results in a dilated net (DRN) block
    '''
    def __init__(self, channels, stride, rate=1, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=1,
            rate=rate,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_1")
        bn_1 = hk.BatchNorm(name=self.name+"_bn_1", **self.bn_config)

        self.layers = ((conv_0, bn_0), (conv_1, bn_1))

    def __call__(self, x, is_training, test_local_stats=False):
        out = x

        for (conv_i, bn_i) in self.layers:
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        return out

class NormTransposeBlock(hk.Module):
    '''
    Normed Conv Transpose Block
    '''
    def __init__(self, channels, stride, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        conv_0 = hk.Conv2DTranspose(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_0")
        bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=1,
            with_bias=False,
            w_init=hk.initializers.Orthogonal(),
            padding="SAME",
            name=self.name+"_conv_1")
        bn_1 = hk.BatchNorm(name=self.name+"_bn_1", **self.bn_config)

        self.layers = ((conv_0, bn_0), (conv_1, bn_1))

    def __call__(self, x, is_training, test_local_stats=False):
        out = x

        for (conv_i, bn_i) in self.layers:
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            out = jax.nn.relu(out)

        return out

class ResNetBlock(hk.Module):
    '''
    Residual Net Block
    Supports dilation via rate that results in a dilated residual net (DRN) block
    '''
    def __init__(self, channels, stride, rate=1, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.proj_conv = hk.Conv2D(
            output_channels=channels,
            kernel_shape=1,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="shortcut_conv")

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.proj_batchnorm = hk.BatchNorm(name=self.name+"_shortcut_bn", **self.bn_config)

        conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_0")
        bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=1,
            rate=rate,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_1")
        bn_1 = hk.BatchNorm(name=self.name+"_bn_1", **self.bn_config)

        self.layers = ((conv_0, bn_0), (conv_1, bn_1))

    def __call__(self, x, is_training, test_local_stats=False):
        out = shortcut = x

        shortcut = self.proj_conv(shortcut)
        shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)
        
class ResNetTransposeBlock(hk.Module):
    '''
    Residual Net Transpose Block
    Supports dilation via rate that results in a dilated residual net (DRN) block
    '''
    def __init__(self, channels, stride, kernel_shape=3, bn_config=None, name=None):
        super().__init__(name=name)

        self.proj_conv = hk.Conv2DTranspose(
            output_channels=channels,
            kernel_shape=1,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="shortcut_conv")

        self.bn_config = dict(bn_config or {})
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)
        self.bn_config.setdefault("decay_rate", 0.999)

        self.proj_batchnorm = hk.BatchNorm(name=self.name+"_shortcut_bn", **self.bn_config)

        conv_0 = hk.Conv2DTranspose(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_0")
        bn_0 = hk.BatchNorm(name=self.name+"_bn_0", **self.bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=kernel_shape,
            stride=1,
            with_bias=False,
            padding="SAME",
            name=self.name+"_conv_1")
        bn_1 = hk.BatchNorm(name=self.name+"_bn_1", **self.bn_config)

        self.layers = ((conv_0, bn_0), (conv_1, bn_1))

    def __call__(self, x, is_training, test_local_stats=False):
        out = shortcut = x

        shortcut = self.proj_conv(shortcut)
        shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)

#dense nets
class MLP(hk.Module):
    '''
    Multi-layer perceptron
    Applies flattening to input and returns logits
    '''
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        '''
        MLP Network
        '''
        #define layers
        net=hk.Flatten()(x)
        net=hk.Linear(256)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(128)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(32)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size)(net)

        return net

#conv nets
class ConvNet(hk.Module):
    '''
    A standard VGG-like network but much smaller
    Assumes image is input and returns logits
    conv_layers defines this many conv layers
    channels sets the initial number of filters
    Subsequent conv layers are increased by powers of 2
    dense_size sets the largest dense layer size right after conv layers
    and uses two dense layers down (each downsampled by 4) to output_size
    '''
    def __init__(self, output_size, conv_layers=2, kernel_shape=3, channels=32, dense_size=256, bn_config=None, logits_config=None, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.conv_layers = conv_layers
        self.kernel_shape = kernel_shape
        self.channels = channels
        self.dense_size = dense_size
        self.bn_config = dict(bn_config or {})

        #batch norm config
        self.bn_config.setdefault("decay_rate", 0.9)
        self.bn_config.setdefault("eps", 1e-5)
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)

        self.logits_config = dict(logits_config or {})
        self.logits_config.setdefault("w_init", jax.numpy.zeros)
        self.logits_config.setdefault("name", "logits")

    def __call__(self, x, is_training, test_local_stats=False):
        '''
        ConvNet Network
        '''
        #define conv layers
        net=NormBlock(self.channels, stride=1, bn_config=self.bn_config, name="normblock_initial_1")(x, is_training)
        net=NormBlock(self.channels, stride=1, bn_config=self.bn_config, name="normblock_initial_2")(net, is_training)
        for n in range(1,self.conv_layers):
            net=NormBlock(self.channels, stride=2, bn_config=self.bn_config, name="normblock_"+str(n)+"_x2")(net, is_training)
            net=NormBlock(self.channels, stride=1, bn_config=self.bn_config, name="normblock_"+str(n)+"_x1")(net, is_training)
        # GlobalAveragePooling2D
        net=jax.numpy.mean(net, axis=[1, 2])
        #define dense layers
        # net=hk.Flatten()(net)
        net=hk.Linear(self.dense_size)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.dense_size//4)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size, **self.logits_config)(net)

        return net

class ResNet(hk.Module):
    '''
    A standard residual network (ResNet 18)
    Assumes image is input and returns logits
    conv_layers defines this many conv layers
    channels sets the initial number of filters
    Subsequent conv layers are increased by powers of 2
    dense_size sets the largest dense layer size right after conv layers
    and uses three dense layers down (each downsampled by 2) to output_size
    '''
    def __init__(self, output_size, channels=64, bn_config=None, initial_conv_config=None, logits_config=None, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.channels = channels
        self.bn_config = dict(bn_config or {})

        #batch norm config
        self.bn_config.setdefault("decay_rate", 0.9)
        self.bn_config.setdefault("eps", 1e-5)
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)

        #initial layer configs
        self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm", **self.bn_config)

        self.initial_conv_config = dict(initial_conv_config or {})
        self.initial_conv_config.setdefault("output_channels", self.channels)
        self.initial_conv_config.setdefault("kernel_shape", 3)
        self.initial_conv_config.setdefault("stride", 1)
        self.initial_conv_config.setdefault("with_bias", False)
        self.initial_conv_config.setdefault("padding", "SAME")
        self.initial_conv_config.setdefault("name", "initial_conv")

        self.initial_conv = hk.Conv2D(**self.initial_conv_config)

        self.logits_config = dict(logits_config or {})
        self.logits_config.setdefault("w_init", jax.numpy.zeros)
        self.logits_config.setdefault("name", "logits")

    def __call__(self, x, is_training, test_local_stats=False):
        '''
        ConvNet Network
        '''
        #define conv layers
        #initial block
        net=self.initial_conv(x)
        net=self.initial_batchnorm(net, is_training, test_local_stats)
        net=jax.nn.relu(net)
        #Res Blocks
        #res layer 1
        net=ResNetBlock(self.channels, stride=1, bn_config=self.bn_config, name="resblock_1a")(net, is_training)
        net=ResNetBlock(self.channels, stride=1, bn_config=self.bn_config, name="resblock_1b")(net, is_training)
        #res layer 2
        net=ResNetBlock(self.channels*2, stride=2, bn_config=self.bn_config, name="resblock_2a")(net, is_training)
        net=ResNetBlock(self.channels*2, stride=1, bn_config=self.bn_config, name="resblock_2b")(net, is_training)
        #res layer 3
        net=ResNetBlock(self.channels*4, stride=2, bn_config=self.bn_config, name="resblock_3a")(net, is_training)
        net=ResNetBlock(self.channels*4, stride=1, bn_config=self.bn_config, name="resblock_3b")(net, is_training)
        #res layer 4
        net=ResNetBlock(self.channels*4, stride=2, bn_config=self.bn_config, name="resblock_4a")(net, is_training)
        net=ResNetBlock(self.channels*4, stride=1, bn_config=self.bn_config, name="resblock_4b")(net, is_training)
        # GlobalAveragePooling2D
        net=jax.numpy.mean(net, axis=[1, 2])
        net=hk.Linear(self.output_size, **self.logits_config)(net)

        return net
