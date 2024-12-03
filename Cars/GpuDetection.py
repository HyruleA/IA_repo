import tensorflow as tf
print("Versión de TensorFlow:", tf.__version__)
print("Dispositivos físicos disponibles:")
print(tf.config.list_physical_devices('GPU'))