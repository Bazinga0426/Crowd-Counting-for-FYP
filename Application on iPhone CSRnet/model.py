""" Here the main body of the csRnet neural network is created"""
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import layers
from keras.initializers import RandomNormal


def CSRNet(input_shape=(None, None, 3)):




    """
    The outlook of the network model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, None, None, 3)     0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, None, None, 64)    1792      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, None, None, 64)    36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, None, None, 64)    0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, None, None, 128)   73856     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, None, None, 128)   147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, None, None, 128)   0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, None, None, 256)   295168    
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, None, None, 256)   590080    
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, None, None, 256)   590080    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, None, None, 256)   0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, None, None, 512)   1180160   
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, None, None, 512)   2359808   
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, None, None, 512)   2359808   
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, None, None, 512)   2359808   
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, None, None, 512)   2359808   
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, None, None, 512)   2359808   
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, None, None, 256)   1179904   
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, None, None, 128)   295040    
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, None, None, 64)    73792     
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, None, None, 1)     65        
    =================================================================
    Total params: 16,263,489
    Trainable params: 16,263,489
    Non-trainable params: 0
    _________________________________________________________________
    
    
    """

    inputs = layers.Input(shape=input_shape)

    # vgg16
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    # Post-Neural Network
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)

    output = layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(x)
    model = Model(inputs=inputs, outputs=output)

    front_end = VGG16(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:  ##  Get the weight of vgg16 specified layer
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 10:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])  ## Set the base weight of vgg16
            counter_conv += 1

    return model



