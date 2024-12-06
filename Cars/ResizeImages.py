import cv2
import os
import numpy as np

def preprocess_images(input_folder, output_folder, size=(21, 28)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Asegúrate de que la imagen sea RGB
            if image is not None and len(image.shape) == 3:
                resized_image = cv2.resize(image, size)
                
                # Normalizar los valores de los píxeles (opcional)
                normalized_image = resized_image / 255.0

                # Guardar la imagen redimensionada (si es necesario)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))  # Si necesitas guardarlas como enteros

    print("Preprocesamiento completo.")

# Especifica las carpetas de entrada y salida
input_folder = "./predataset/audi"
output_folder = "./dataset/audi"

# Llama a la función
preprocess_images(input_folder, output_folder, size=(21, 28))
