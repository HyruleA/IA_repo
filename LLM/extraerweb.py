import requests
from bs4 import BeautifulSoup

pagina = "https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S2448-57052022000200315"
destino = "./web/ORGAUT.txt"

respuesta = requests.get(pagina, verify=False)
if respuesta.ok:
    contenido = respuesta.content
    analizador = BeautifulSoup(contenido, "html.parser")
    parrafos = [etiqueta.get_text(strip=True) for etiqueta in analizador.find_all("p")]
    with open(destino, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(parrafos))
    print("Texto guardado")
else:
    print("No fue posible obtener el contenido.")  
