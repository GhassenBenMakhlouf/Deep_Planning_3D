from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from data import DP3dDataset
from models import build_cnn_3d, build_cnn_3d_cheap
import tensorflow as tf
import tensorflow_addons as tfa
import random
import datetime
import time


def train_cnn_3d(config):
    cfg = config.learning
    train_path = cfg.trainset_path
    val_path = cfg.valset_path
    test_path = cfg.testset_path
    batch_size = cfg.batchsize
    number_of_epochs = cfg.n_epochs

    physical_devices = tf.config.list_physical_devices("GPU")
    print("Num GPUs:", len(physical_devices))

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    train_dataset = DP3dDataset(dataset_path=train_path, batch_size=batch_size)
    val_dataset = DP3dDataset(dataset_path=val_path, batch_size=batch_size)
    test_dataset = DP3dDataset(dataset_path=test_path, batch_size=batch_size)
    input_shape = train_dataset[0][0].shape[1:]

    print("input_shape: ", input_shape)

    print(
        "(trainsize, valsize, testsize) =",
        len(train_dataset),
        " ",
        len(val_dataset),
        " ",
        len(test_dataset),
    )

    if cfg.cheap:
        model = build_cnn_3d_cheap(input_shape=input_shape)
    else:
        model = build_cnn_3d(input_shape=input_shape)
    # model.build((None, 101, 101, 101, 2))
    # print('#parameters:', model.count_params()/1e6, 'm')

    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredLogarithmicError())
    print("#parameters:", model.count_params())

    x = tf.ones((2, 101, 101, 101))

    t = time.time()
    # y = model(tf.expand_dims(X[0], axis=0))
    y = model(x)
    elapsed = time.time() - t
    print(elapsed, "s to run forward pass")
    print(y.shape)

    model.summary()

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

    checkpoint_path = "checkpoints/" + date_time + "/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch",
        period=100,
    )
    callbacks = [
        reduce_lr,
        early_stopping,
        tqdm_callback,
        tensorboard_callback,
        cp_callback,
    ]

    random.seed(42)
    tf.random.set_seed(42)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=number_of_epochs,
        callbacks=callbacks,
        verbose=3,
    )
