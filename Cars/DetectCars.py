import os
import subprocess

def process_images_with_yolo(input_folder, car, output_folder, conf_threshold=0.5, classes="2"):
    # Verificar que las carpetas existan
    if not os.path.exists(input_folder):
        raise ValueError(f"La carpeta de entrada '{input_folder}' no existe.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ruta al intérprete de Python del entorno virtual, apuntarlo al python.exe dentro de su propio entorno virtual con las librerias instaladas
    python_executable = r"C:/Proyectos/IA_repo/env/Scripts/python.exe"

    # Iterar sobre todas las imágenes en el directorio de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
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
            print(f"Procesando: {filename}")
            # Ejecutar el comando YOLO
            subprocess.run(command)

    print(f"Todas las imágenes se han procesado. Resultados guardados en '{output_folder}'.")
    
# Especifica las carpetas de entrada y salida

car = "f150"
folder = "predataset"
input_folder = folder+"/"+car
# Llama a la función
process_images_with_yolo(input_folder, car, folder, conf_threshold=0.5, classes="2")
