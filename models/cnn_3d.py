from keras.models import Sequential
from keras.layers import Dropout, Conv3D, BatchNormalization, MaxPool3D, Conv3DTranspose


def build_cnn_3d(input_shape):
    model = Sequential()
    # encoder
        
    model.add(Conv3D(filters=16, kernel_size=11, strides=(1, 1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool3D(pool_size=(1, 1, 1), strides=(2,2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 51x51x51
    model.add(Conv3D(filters=32, kernel_size=7, strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(MaxPool3D(pool_size=(1, 1, 1), strides=(2,2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 26x26x26
    model.add(Conv3D(filters=64, kernel_size=5, strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(MaxPool3D(pool_size=(1, 1, 1), strides=(2,2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 13x13x13
    model.add(Conv3D(filters=128, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(MaxPool3D(pool_size=(1, 1, 1), strides=(2,2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 7x7x7
    model.add(Conv3D(filters=256, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(MaxPool3D(pool_size=(1, 1, 1), strides=(2,2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 4x4x4


    # decoder
       
    model.add(Conv3DTranspose(filters=256, kernel_size=3, strides=(2, 2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 8x8x8

    model.add(Conv3DTranspose(filters=128, kernel_size=3, strides=(2, 2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 16x16x16

    model.add(Conv3DTranspose(filters=64, kernel_size=5, strides=(2, 2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 32x32x32

    model.add(Conv3DTranspose(filters=32, kernel_size=7, strides=(1, 1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 38x38x38

    model.add(Conv3DTranspose(filters=16, kernel_size=9, strides=(1, 1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 46x46x46

    model.add(Conv3DTranspose(filters=8, kernel_size=11, strides=(2, 2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # the scene is now 101x101x101    

    model.add(Conv3D(filters=2, kernel_size=1, strides=(1, 1, 1), padding='same', activation='relu'))
    
    return model   
