import os
import shutil

def unify_and_rename(folder1, folder2, output_folder):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Inicializar el contador para los nombres
    counter = 1

    # Procesar ambas carpetas
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            # Generar un nuevo nombre con un contador consecutivo
            new_name = f"image_{counter:04d}{os.path.splitext(filename)[1]}"
            counter += 1

            # Mover el archivo a la carpeta de salida con el nuevo nombre
            source_path = os.path.join(folder, filename)
            destination_path = os.path.join(output_folder, new_name)
            shutil.copy(source_path, destination_path)

    print(f"Todas las imágenes han sido unificadas en '{output_folder}' con nombres consecutivos.")

# Especifica las rutas de las carpetas
folder1 = "dataset/nissan2016/nissan march 2016"
folder2 = "dataset/nissan2017/nissan march 2017"
output_folder = "dataset/march"

# Llamar a la función
unify_and_rename(folder1, folder2, output_folder)
