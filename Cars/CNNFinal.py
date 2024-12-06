import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import classification_report

# Ruta base del dataset
imgpath = os.path.join(os.getcwd(), 'DataAug') + os.sep

images = []
labels = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("Leyendo imágenes de", imgpath)
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        # Evitar el warning de escape sequence usando r""
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename.lower()):
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)  # Lee la imagen
            # Sólo consideramos imágenes con 3 canales
            if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
                images.append(image)
                # Si cambió el directorio, contabilizar
                if prevRoot != root:
                    directories.append(root)
                    dircount.append(cant)
                    cant = 0
                    prevRoot = root
                cant += 1
dircount.append(cant)
dircount = dircount[1:]
dircount[0] = dircount[0] + 1

print('Directorios leídos:', len(directories))
print("Imágenes en cada directorio", dircount)
print('Suma total de imágenes en subdirs:', sum(dircount))

# Crear etiquetas según la cantidad de imágenes por directorio
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice+1

# Obtener los nombres de las clases
sriesgos = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    sriesgos.append(name[-1])
    indice += 1

y = np.array(labels)
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Ahora dividimos el dataset manualmente (ya que las imágenes tienen tamaños diferentes)
num_images = len(images)
indices = np.arange(num_images)
np.random.shuffle(indices)

train_end = int(0.6 * num_images)
valid_end = int(0.8 * num_images)

train_indices = indices[:train_end]
valid_indices = indices[train_end:valid_end]
test_indices = indices[valid_end:]

train_X = [images[i] for i in train_indices]
train_Y = [y[i] for i in train_indices]

valid_X = [images[i] for i in valid_indices]
valid_Y = [y[i] for i in valid_indices]

test_X = [images[i] for i in test_indices]
test_Y = [y[i] for i in test_indices]

# Normalización a [0,1] se hace on-the-fly en el pipeline.  
# Aquí simplemente dejamos las imágenes como están en uint8 o el tipo original.

# One-hot encoding de las etiquetas
train_Y_one_hot = to_categorical(train_Y)
valid_Y_one_hot = to_categorical(valid_Y)
test_Y_one_hot = to_categorical(test_Y)

print("Tamaños:", len(train_X), len(valid_X), len(test_X))

# Parámetros de entrenamiento
INIT_LR = 1e-3
epochs = 30
batch_size = 32
target_height, target_width = 80, 80

# Función para resize con cv2
def resize_with_cv2(image):
    # image es un tensor, lo convertimos a numpy
    image_np = image.numpy()  # [H,W,3] float32
    # Resize a (21,28)
    resized = cv2.resize(image_np, (target_width, target_height))  # (ancho, alto)
    return resized.astype(np.float32)

def augment_and_resize(image, label):
    # Convertir a float32 [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Aumento de datos
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Resize con cv2
    image = tf.py_function(resize_with_cv2, [image], tf.float32)
    image = tf.reshape(image, [target_height, target_width, 3])
    return image, label

# Generador para el dataset
def gen_data(X_list, Y_array):
    # X_list es una lista de imágenes de diferentes tamaños
    # Y_array son las etiquetas one-hot
    for x, y in zip(X_list, Y_array):
        yield x, y

# Creamos los datasets con from_generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: gen_data(train_X, train_Y_one_hot),
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(nClasses,), dtype=tf.float32)
    )
)
valid_dataset = tf.data.Dataset.from_generator(
    lambda: gen_data(valid_X, valid_Y_one_hot),
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(nClasses,), dtype=tf.float32)
    )
)
test_dataset = tf.data.Dataset.from_generator(
    lambda: gen_data(test_X, test_Y_one_hot),
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(nClasses,), dtype=tf.float32)
    )
)

# Aplicar augment_and_resize sólo al train
train_dataset = train_dataset.map(augment_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Para validación y test, sólo resize (sin aumentos)
def just_resize(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.py_function(resize_with_cv2, [image], tf.float32)
    image = tf.reshape(image, [target_height, target_width, 3])
    return image, label

valid_dataset = valid_dataset.map(just_resize, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(just_resize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Definir el modelo
model = Sequential()

# Bloque 1
model.add(Conv2D(32, (3,3), padding='same', input_shape=(80,80,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Bloque 2
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

'''# Bloque 3 (opcional, para mayor profundidad)
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))'''

# Cambiar Flatten por GlobalAveragePooling2D
model.add(GlobalAveragePooling2D())

# Capas densas
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# Entrenamiento
history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, verbose=1)

# Evaluación
test_loss, test_acc = model.evaluate(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Visualización de curvas
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(accuracy))
plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predicción
predicted_classes2 = model.predict(test_dataset)
predicted_classes=[]
for predicted_riesgo in predicted_classes2:
    predicted_classes.append(predicted_riesgo.tolist().index(max(predicted_riesgo)))
predicted_classes=np.array(predicted_classes)

print(classification_report(test_Y, predicted_classes, target_names=["Class {}".format(i) for i in range(nClasses)]))
