import os
import subprocess
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Ruta al modelo preentrenado
model_path = './NN/ModelDensoFinalChido.keras'
model = load_model(model_path)

# Mapeo de clases
class_labels = {
    0: "Audi R8",
    1: "Chevrolet Venture",
    2: "Honda Civic",
    3: "Nissan March",
    4: "Ford Mustang"
}

def process_image_with_yolo(input_path, output_folder, car, conf_threshold=0.5, classes="2"):
    # Ruta al intérprete de Python del entorno virtual
    python_executable = r"C:/Proyectos/IA_repo/env/Scripts/python.exe"
    
    # Verificar que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Definir el comando YOLO
    command = [
        python_executable, "yolov5/detect.py",  # Ruta al script detect.py
        "--source", input_path,  # Imagen de entrada
        "--classes", classes,  # ID de la clase a detectar
        "--save-crop",  # Guardar los recortes
        "--conf-thres", str(conf_threshold),  # Umbral de confianza
        "--project", output_folder,  # Carpeta de salida
        "--name", car,  # Subcarpeta de salida
        "--exist-ok",  # Sobreescribir si ya existe
        "--nosave"
    ]
    print(f"Procesando con YOLO: {input_path}")
    
    # Ejecutar el comando YOLO
    subprocess.run(command)

    # Determinar el subdirectorio de salida de YOLO
    output_yolo_folder = os.path.join(output_folder, car, "crops")
    
    # Buscar la primera imagen procesada (recorte)
    if os.path.exists(output_yolo_folder):
        for root, dirs, files in os.walk(output_yolo_folder):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    return os.path.join(root, file)
    
    raise ValueError("No se encontró una imagen procesada en la salida de YOLO.")

def preprocess_image(image_path, target_size=(80, 80)):
    """
    Redimensiona y normaliza una imagen para que sea compatible con el modelo.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Asegurarse de que la imagen sea RGB
    if image is None or len(image.shape) != 3:
        raise ValueError(f"La imagen no es válida o no es RGB: {image_path}")

    # Redimensionar la imagen
    resized_image = cv2.resize(image, target_size)

    # Normalizar los valores de los píxeles (0-1)
    normalized_image = resized_image / 255.0

    # Expandir dimensiones para el modelo
    image_batch = np.expand_dims(normalized_image, axis=0)

    return image_batch

def predict_image(input_path, car, yolo_output_folder, conf_threshold=0.5, classes="2"):
    """
    Procesa una imagen con YOLO, luego la redimensiona y predice con el modelo.
    """
    # Procesar la imagen con YOLO
    cropped_image_path = process_image_with_yolo(input_path, yolo_output_folder, car, conf_threshold, classes)
    
    # Preprocesar la imagen recortada
    preprocessed_image = preprocess_image(cropped_image_path)

    # Realizar la predicción
    predictions = model.predict(preprocessed_image)

    # Obtener la clase y la confianza
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Etiqueta de clase
    class_label = class_labels.get(predicted_class, "Clase desconocida")

    return class_label, confidence

# Ejemplo de uso
car = ""
folder = "."
input_folder = folder + "/" + car
image_path = input_folder + "mustang2.jpg"  # Cambia por tu imagen

try:
    predicted_label, confidence = predict_image(image_path, car, folder, conf_threshold=0.5, classes="2")
    print(f"Predicción: {predicted_label} con una confianza de {confidence:.2f}")
except Exception as e:
    print(f"Error al procesar: {e}")
