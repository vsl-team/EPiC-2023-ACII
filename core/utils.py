import tensorflow as tf
import keras.utils as kr_utils


def set_seed(seed=1):
    kr_utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()


def set_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            n_gpus = len(gpus)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            n_gpus = 0
            print(e)
    else:
        n_gpus = 0

    return n_gpus


def set_mixed_precision(mixed_precision=True):
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision training')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print('Float32 training')
