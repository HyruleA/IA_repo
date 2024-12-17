import re
import os

def limpiar_texto(texto):
    texto = re.sub(r'\n+', '\n', texto)
    texto = re.sub(r'(Página\s\d+)', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

entrada = "./PDFsExtraido"
salida = "./txtDocsyWeb"
os.makedirs(salida, exist_ok=True)

for archivo in os.listdir(entrada):
    if archivo.lower().endswith(".txt"):
        ruta_entrada = os.path.join(entrada, archivo)
        with open(ruta_entrada, "r", encoding="utf-8") as origen:
            texto_original = origen.read()

        texto_limpio = limpiar_texto(texto_original)
        ruta_salida = os.path.join(salida, archivo)

        with open(ruta_salida, "w", encoding="utf-8") as destino:
            destino.write(texto_limpio)

        print(f"Se ha procesado y guardado el texto en: {ruta_salida}")
