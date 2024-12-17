import pdfplumber
import os

entrada = "./PDFs"
salida = "./PDFsExtraido"
os.makedirs(salida, exist_ok=True)

for archivo in os.listdir(entrada):
    if archivo.lower().endswith(".pdf"):
        ruta_pdf = os.path.join(entrada, archivo)
        nombre_sin_ext = os.path.splitext(archivo)[0]
        ruta_txt = os.path.join(salida, f"{nombre_sin_ext}.txt")

        with pdfplumber.open(ruta_pdf) as pdf:
            texto_extraido = "\n".join(pagina.extract_text() for pagina in pdf.pages if pagina.extract_text())

        with open(ruta_txt, "w", encoding="utf-8") as destino:
            destino.write(texto_extraido)

        print(f"Archivo procesado y texto guardado en: {ruta_txt}")
