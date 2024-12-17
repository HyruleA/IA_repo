import os
import json

carpeta_txt = "./txtDocsyWeb"
datos = []

if not os.path.isdir(carpeta_txt):
    raise FileNotFoundError(f"No se encontró la carpeta: {carpeta_txt}")

for archivo in os.listdir(carpeta_txt):
    if archivo.lower().endswith(".txt"):
        ruta_txt = os.path.join(carpeta_txt, archivo)
        with open(ruta_txt, "r", encoding="utf-8") as contenido:
            texto = contenido.read()
            datos.append({
                "prompt": f"Contenido de {archivo}",
                "response": texto
            })

archivo_salida = "./dataset.jsonl"
with open(archivo_salida, "w", encoding="utf-8") as salida:
    for item in datos:
        json_linea = json.dumps(item, ensure_ascii=False)
        salida.write(json_linea + "\n")

print(f"Archivo generado: {archivo_salida}")
