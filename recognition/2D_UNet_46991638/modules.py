import tensorflow as tf 
#Modified from https://github.com/shakes76/PatternFlow/blob/master/recognition/MySolution/Methods.ipynb
def unet_model ():    
    filter_size=16 
    input_layer = tf.keras.Input((256,256,1))
    
    pre_conv = tf.keras.layers.Conv2D(filter_size * 1, (3, 3), padding="same")(input_layer)
    pre_conv = tf.keras.layers.LeakyReLU(alpha=.01)(pre_conv)


# context module 1 pre-activation residual block
    conv1 = tf.keras.layers.BatchNormalization()(pre_conv)
    conv1 = tf.keras.layers.LeakyReLU(alpha=.01)(conv1)
    conv1 = tf.keras.layers.Conv2D(filter_size * 1, (3, 3), padding="same" )(conv1) 
    conv1 = tf.keras.layers.Dropout(.3) (conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.LeakyReLU(alpha=.01)(conv1)
    conv1 = tf.keras.layers.Conv2D(filter_size * 1, (3, 3), padding="same")(conv1)    
    conv1 = tf.keras.layers.Add()([pre_conv,conv1])
    
# downsample and double number of feature maps   
    pool1 = tf.keras.layers.Conv2D(filter_size * 2, (3,3), (2,2) , padding='same')(conv1)
    pool1 = tf.keras.layers.LeakyReLU(alpha=.01)(pool1)
    
# context module 2
    conv2 = tf.keras.layers.BatchNormalization()(pool1)
    conv2 = tf.keras.layers.LeakyReLU(alpha=.01)(conv2)
    conv2 = tf.keras.layers.Conv2D(filter_size * 2, (3, 3), padding="same")(conv2)
    conv2 = tf.keras.layers.Dropout(.3) (conv2)  
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.LeakyReLU(alpha=.01)(conv2)
    conv2 = tf.keras.layers.Conv2D(filter_size * 2, (3, 3), padding="same")(conv2)
    conv2 = tf.keras.layers.Add()([pool1,conv2])

# downsample and double number of feature maps
    pool2 = tf.keras.layers.Conv2D(filter_size*4, (3,3),(2,2), padding='same')(conv2)
    pool2 = tf.keras.layers.LeakyReLU(alpha=.01)(pool2)

# context module 3
    conv3 = tf.keras.layers.BatchNormalization()(pool2)
    conv3 = tf.keras.layers.LeakyReLU(alpha=.01)(conv3)
    conv3 = tf.keras.layers.Conv2D(filter_size * 4, (3, 3), padding="same")(conv3)
    conv3 = tf.keras.layers.Dropout(.3) (conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.LeakyReLU(alpha=.01)(conv3)
    conv3 = tf.keras.layers.Conv2D(filter_size * 4, (3, 3), padding="same")(conv3)
    conv3 = tf.keras.layers.Add()([pool2,conv3])

# downsample and double number of feature maps
    pool3 = tf.keras.layers.Conv2D(filter_size*8, (3,3),(2,2),padding='same')(conv3)
    pool3 = tf.keras.layers.LeakyReLU(alpha=.01)(pool3)

# context module 4
    conv4 = tf.keras.layers.BatchNormalization()(pool3)
    conv4 = tf.keras.layers.LeakyReLU(alpha=.01)(conv4)
    conv4 = tf.keras.layers.Conv2D(filter_size * 8, (3, 3), padding="same")(conv4)
    conv4 = tf.keras.layers.Dropout(.3) (conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.LeakyReLU(alpha=.01)(conv4)
    conv4 = tf.keras.layers.Conv2D(filter_size * 8, (3, 3), padding="same")(conv4)
    conv4 = tf.keras.layers.Add()([pool3,conv4])
    

# downsample and double number of feature maps
    pool4 = tf.keras.layers.Conv2D(filter_size*16, (3,3),(2,2),padding='same')(conv4)
    pool4 = tf.keras.layers.LeakyReLU(alpha=.01)(pool4) 

# context module 5
    # Middle
    convm = tf.keras.layers.BatchNormalization()(pool4)
    convm = tf.keras.layers.LeakyReLU(alpha=.01)(convm)
    convm = tf.keras.layers.Conv2D(filter_size * 16, (3, 3), padding="same")(convm)
    convm = tf.keras.layers.Dropout(.3) (convm)
    convm = tf.keras.layers.BatchNormalization()(convm)
    convm = tf.keras.layers.LeakyReLU(alpha=.01)(convm)
    convm = tf.keras.layers.Conv2D(filter_size * 16, (3, 3), padding="same")(convm)
    convm = tf.keras.layers.Add()([pool4,convm])


#upsampling module 1
    deconv4 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(convm)
    deconv4 = tf.keras.layers.Conv2D (filter_size *8, (3, 3) , padding="same")(deconv4)
    deconv4 = tf.keras.layers.LeakyReLU(alpha=.01)(deconv4) 
    

#concatatinate layers 
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4], axis=3)


#localization module 1
    uconv4 = tf.keras.layers.Conv2D(filter_size * 16, (3, 3) , padding="same")(uconv4)
    uconv4 = tf.keras.layers.BatchNormalization()(uconv4)
    uconv4 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv4)
    uconv4 = tf.keras.layers.Conv2D(filter_size * 8, (1, 1), padding="same")(uconv4)
    uconv4 = tf.keras.layers.BatchNormalization()(uconv4)
    uconv4 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv4)

#upsampling module 2
    deconv3 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(uconv4)
    deconv3 = tf.keras.layers.Conv2D (filter_size *4, (3, 3) , padding="same")(deconv3)
    deconv3 = tf.keras.layers.LeakyReLU(alpha=.01)(deconv3) 

  

# concatatinate layers  
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3], axis=3)


# localization module 2
    uconv3 = tf.keras.layers.Conv2D(filter_size * 8, (3, 3), padding="same")(uconv3)
    uconv3 = tf.keras.layers.BatchNormalization()(uconv3)
    uconv3 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv3)
    uconv3 = tf.keras.layers.Conv2D(filter_size * 4, (1, 1), padding="same")(uconv3)
    uconv3 = tf.keras.layers.BatchNormalization()(uconv3)
    uconv3 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv3)

# segmentation layer 1
    seg3 = tf.keras.layers.Conv2D(4, (3,3),  activation="softmax", padding='same')(uconv3)
# upscale segmented layer 1
    seg3 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(seg3)


# Upsample module 3
    deconv2 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(uconv3)
    deconv2 = tf.keras.layers.Conv2D (filter_size *2, (3, 3) , padding="same")(deconv2)
    deconv2 = tf.keras.layers.LeakyReLU(alpha=.01)(deconv2)


# concatination layer 
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2], axis=3)


# localization module 3
    uconv2 = tf.keras.layers.Conv2D(filter_size * 4, (3, 3), padding="same")(uconv2)
    uconv2 = tf.keras.layers.BatchNormalization()(uconv2)
    uconv2 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(filter_size * 2, (1, 1), padding="same")(uconv2)
    uconv2 = tf.keras.layers.BatchNormalization()(uconv2)
    uconv2 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv2)

# segmentation layer 2
    seg2 = tf.keras.layers.Conv2D(4, (3,3),  activation="softmax", padding='same')(uconv2)

# add segmentation layer 1 and 2
    seg_32 = tf.keras.layers.Add()([seg3,seg2])
# upscale sum segmentation layer 1 and 2
    seg_32 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(seg_32)


# Upsample module 4
    deconv1 = tf.keras.layers.UpSampling2D(size=(2,2) , interpolation='bilinear')(uconv2)
    deconv1 = tf.keras.layers.Conv2D (filter_size *1, (3, 3) , padding="same")(deconv1)
    deconv1 = tf.keras.layers.LeakyReLU(alpha=.01)(deconv1)


# concatination layer
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1], axis=3 )

#final convolution layer
    uconv1 = tf.keras.layers.Conv2D(filter_size * 2, (3, 3), padding="same")(uconv1)
    uconv1 = tf.keras.layers.BatchNormalization()(uconv1)
    uconv1 = tf.keras.layers.LeakyReLU(alpha=.01)(uconv1)
    
# final segmentation layer   
    seg1 = tf.keras.layers.Conv2D(4, (3,3),  activation="softmax", padding='same' )(uconv1)

# sum all segmentation layers 
    seg_sum = tf.keras.layers.Add()([seg1,seg_32])


    output_layer = tf.keras.layers.Conv2D(6, (3,3), padding='same' ,activation="softmax")(seg_sum)
    model = tf.keras.Model( input_layer , outputs=output_layer)
    return model
