# Contains components of model.
import keras
from keras import layers, models, Sequential
from keras.src.backend import shape
from keras.src.losses import Dice
from keras.src.optimizers import AdamW
from keras.src import backend
from keras.src import ops


FULL_SIZE_IMG = 1  # set to 2 to use full size image
INPUT_SHAPE = (32, 32, 3)
num_classes = 4  # numb of classes in segmentation

def dice_loss(y_true, y_pred, axis=None):
    # this is the Dice() code
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    inputs = y_true
    targets = y_pred

    intersection = ops.sum(inputs * targets, axis=axis)
    dice = ops.divide(
        2.0 * intersection,
        ops.sum(y_true, axis=axis)
        + ops.sum(y_pred, axis=axis)
        + backend.epsilon(),
    )

    return 1 - dice

def unet_model(input_size=(128, 128, 1), batch_size=12, preprocessing=None):
    keras.backend.clear_session()

    inputs = layers.Input(input_size)


    # flattened = layers.Flatten(inputs)

    # rand_crop = layers.RandomCrop()(inputs)
    rand_flip = layers.RandomFlip(mode="horizontal_and_vertical")(inputs)

    norm = layers.Normalization()(rand_flip)

    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(norm)  # padding same accounts for the shrinkage that occurs from kernal
    dropped = layers.Dropout(0.25)(conv1)
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(dropped)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    # conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    # pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)


    # Bridge
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    # up1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    # concat1 = layers.concatenate([up1, conv4], axis=3)
    # conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    # conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    print("Conv5 shape: "+str(shape(conv5)))
    print("Conv3 shape: " + str(shape(conv3)))
    print("Conv2 shape: " + str(shape(conv2)))
    print("Conv1 shape: " + str(shape(conv1)))

    up2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)  # Changed conv 6 to conv 5
    concat2 = layers.concatenate([up2, conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat3 = layers.concatenate([up3, conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8) # Changed 8 to 5
    concat4 = layers.concatenate([up4, conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(6, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer=AdamW(learning_rate=0.0001), loss=Dice, metrics=['accuracy'])
    # keras_cv.losses.IoULoss("xyxy", mode="quadratic")

    return model

# def dice_coef(y_true, y_pred, smooth=1):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf
#     """
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
#
# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef(y_true, y_pred)


# read: https://keras.io/examples/vision/oxford_pets_image_segmentation/

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# '''
# Network Summary: https://medium.com/analytics-vidhya/what-is-unet-157314c87634
#
# Encoder Blocks:
# - each block consists of two 3 x 3 convolutions.
# - each conv is are followed by a ReLU.
# - The output of the ReLU acts as a skip connecttion for the corresponding decoder block
#
# - Next 2x2 max pooling halves the dims of the feature map.
#
# Bridge:
# - Two 3x3 convs, where each is followed by a ReLU
#
# Decoder Blocks:
# - used to take abstract representation and generate a semantic segmentation mask.
# - Starts with 2x2 transpose convolution
# - ^ is concatinated with skip connection feature map from the corresponding encoder block'
# - two 3x3 convolutions are used. Each is followed by ReLUs.
#
# Useful:
# - ConvTranspose2d
# - MaxPool2d
#
# '''
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_pixels, out_pixels, middle_pixels, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_pixels, middle_pixels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(middle_pixels, out_pixels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv_block(x)
#
#
# class Encode(nn.Module):
#     def __init__(self, in_pixels, out_pixels):
#         super().__init__()
#
#         self.pool_and_conv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2),
#             ConvBlock(in_pixels, out_pixels, out_pixels)
#         )
#
#     def forward(self, x):
#         return self.pool_and_conv(x)
#
#
# class Decode(nn.Module):
#     def __init__(self, in_pixels, out_pixels):
#         super().__init__()
#
#         self.upsamp = nn.Upsample(2, mode='bilinear', align_corners=False)
#         self.conv_block = ConvBlock(in_pixels, out_pixels, out_pixels)
#
#     def forward(self, x, skip):
#         print(f"x before upsampling: {x.size()}")
#         x = self.upsamp(x)
#         print(f"x after upsampling: {x.size()}")
#
#         # Calculate padding needed
#         diff_x = skip.size()[2] - x.size()[2]#
# #
# # # ============== Vars ==============
# #
# # FULL_SIZE_IMG = 1  # set to 2 to use full size image
# # INPUT_SHAPE = (32, 32, 3)
# # num_classes = 4  # numb of classes in segmentation
# #
# # # ============== Get Data ==============
# #
# # def load_and_preprocess_image(image_path, target_size):
# #     """Load an image, resize it, and normalize it."""
# #     image = load_img(image_path, target_size=target_size, color_mode='grayscale')
# #     image = img_to_array(image) / 255.0  # Normalize to [0, 1]
# #     return image
# #
# #
# # def load_data(image_folder, mask_folder, target_size):
# #     """Load and preprocess images and masks."""
# #     # sorted takes in an array.
# #     # the for loop creates an array with all the png names in a folder.
# #     image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
# #     mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
# #
# #     images = []
# #     masks = []
# #
# #     for img_name, mask_name in zip(image_filenames, mask_filenames):
# #         img_path = os.path.join(image_folder, img_name)
# #         mask_path = os.path.join(mask_folder, mask_name)
# #
# #         image = load_and_preprocess_image(img_path, target_size)
# #         mask = load_and_preprocess_image(mask_path, target_size)
# #
# #         images.append(image)
# #         masks.append(mask)
# #
# #     return np.array(images), np.array(masks)
# #
# # # Set the target size of the images
# #
# # target_size = (128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG)
# #
# # # training data
# # train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)
# #
# # # testing data
# # test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)
# #
# # # For binary masks
# # train_masks = train_masks.astype(np.float32)
# # test_masks = test_masks.astype(np.float32)
# #
# # # Convert masks to categorical one-hot encodings
# # train_masks = to_categorical(train_masks, num_classes=num_classes)
# # test_masks = to_categorical(test_masks, num_classes=num_classes)
# #
#
#         diff_y = skip.size()[3] - x.size()[3]
#         print(f"Padding to be applied: {[diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]}")
#
#         x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
#         print(f"x after padding: {x.size()}")
#
#         x_merged = torch.cat([skip, x], dim=1)
#         print(f"x_merged after concatenation: {x_merged.size()}")
#
#         return self.conv_block(x_merged)
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),
#             # nn.Sigmoid()
#         )#
# #
# # # ============== Vars ==============
# #
# # FULL_SIZE_IMG = 1  # set to 2 to use full size image
# # INPUT_SHAPE = (32, 32, 3)
# # num_classes = 4  # numb of classes in segmentation
# #
# # # ============== Get Data ==============
# #
# # def load_and_preprocess_image(image_path, target_size):
# #     """Load an image, resize it, and normalize it."""
# #     image = load_img(image_path, target_size=target_size, color_mode='grayscale')
# #     image = img_to_array(image) / 255.0  # Normalize to [0, 1]
# #     return image
# #
# #
# # def load_data(image_folder, mask_folder, target_size):
# #     """Load and preprocess images and masks."""
# #     # sorted takes in an array.
# #     # the for loop creates an array with all the png names in a folder.
# #     image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
# #     mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
# #
# #     images = []
# #     masks = []
# #
# #     for img_name, mask_name in zip(image_filenames, mask_filenames):
# #         img_path = os.path.join(image_folder, img_name)
# #         mask_path = os.path.join(mask_folder, mask_name)
# #
# #         image = load_and_preprocess_image(img_path, target_size)
# #         mask = load_and_preprocess_image(mask_path, target_size)
# #
# #         images.append(image)
# #         masks.append(mask)
# #
# #     return np.array(images), np.array(masks)
# #
# # # Set the target size of the images
# #
# # target_size = (128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG)
# #
# # # training data
# # train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)
# #
# # # testing data
# # test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)
# #
# # # For binary masks
# # train_masks = train_masks.astype(np.float32)
# # test_masks = test_masks.astype(np.float32)
# #
# # # Convert masks to categorical one-hot encodings
# # train_masks = to_categorical(train_masks, num_classes=num_classes)
# # test_masks = to_categorical(test_masks, num_classes=num_classes)
# #
#
#
#     def forward(self, x):
#         return self.out_conv(x)
#
#
# class UNet(nn.Module):
#     def __init__(self, num_channels, num_classes):
#         super().__init__()
#
#         self.num_channels = num_channels
#         self.num_classes = num_classes
#
#         self.first = ConvBlock(num_channels, 64, 64)
#         self.encode1 = Encode(64, 128)
#         self.encode2 = Encode(128, 256)
#         self.encode3 = Encode(256, 512)
#         self.bridge = Encode(512, 512)
#         self.decode1 = Decode(512, 256)
#         self.decode2 = Decode(256, 128)
#         self.decode3 = Decode(128, 64)
#         self.decode4 = Decode(64, 32)
#         self.out = OutConv(32, num_classes)
#
#     def forward(self, x):
#         print('iteration')
#         x1 = self.first(x)
#         x2 = self.encode1(x1)
#         x3 = self.encode2(x2)
#         x4 = self.encode3(x3)
#         x5 = self.bridge(x4)
#
#         x = self.decode1(x5, x4)
#         x = self.decode2(x, x3)
#         x = self.decode3(x, x2)
#         x = self.decode4(x, x1)
#         out = self.out(x)
#         print('iteration end')
#         return out
#
#
# TRansforms to include: Rescale, RandomCrop, Compose