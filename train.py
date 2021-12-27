from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
import datetime
import os
from keras.models import Model, Sequential, load_model
from keras.layers import Dropout, Conv3D, BatchNormalization, MaxPool3D, Conv3DTranspose
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import regularizers, losses

from data import DP3dDataset
from models import build_cnn_3d


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    print("Num GPUs:", len(physical_devices))

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    train_path = "./numpydata_training"
    val_path = "./numpydata_validation"
    test_path = "./numpydata_testing"
    batch_size = 14

    # X, Y = load_dataset(dataset_path=dataset_path)
    train_dataset = DP3dDataset(dataset_path=train_path, batch_size=batch_size)
    val_dataset = DP3dDataset(dataset_path=val_path, batch_size=batch_size)
    test_dataset = DP3dDataset(dataset_path=test_path, batch_size=batch_size)
    input_shape = train_dataset[0][0].shape[1:]
    print("input_shape: ", input_shape)

    # X_np_train, X_np_test, Y_np_train, Y_np_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    # print('(trainsize, testsize) =', len(X_np_train), ' ', len(X_np_test))
    print(
        "(trainsize, valsize, testsize) =",
        len(train_dataset),
        " ",
        len(val_dataset),
        " ",
        len(test_dataset),
    )

    model = build_cnn_3d(input_shape=input_shape)

    # model.build((None, 101, 101, 101, 2))
    # print('#parameters:', model.count_params()/1e6, 'm')
    # print('#parameters decoder:', (model.count_params()-2698784)/1e6, 'm')

    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredLogarithmicError())
    print("#parameters:", model.count_params())

    # x = tf.ones((2,101,101,101,2))

    # t = time.time()
    # y = model(tf.expand_dims(X[0], axis=0))
    # elapsed = time.time() - t
    # print(elapsed, "s to run forward pass")
    # print(y.shape)

    # model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=100, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.75, patience=20, min_lr=0.00003
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar()

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + date_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    checkpoint_path = "checkpoints/"+ date_time +"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq="epoch", period=100
    )
    callbacks = [
        reduce_lr,
        early_stopping,
        tqdm_callback,
        tensorboard_callback,
        cp_callback,
    ]

    number_of_epochs = 10000

    random.seed(42)
    tf.random.set_seed(42)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # history = model.fit(X_np_train, Y_np_train,
    #                     epochs=number_of_epochs, batch_size=batch_size, validation_split=0.10,
    #                     callbacks=callbacks, verbose=3)

    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=number_of_epochs,
        callbacks=callbacks,
        verbose=3,
    )
