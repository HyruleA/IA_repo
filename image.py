from PIL import Image

# Abre la imagen
img = Image.open('C:/Users/Link_/Desktop/20x20.png')

# Asegúrate de que la imagen esté en modo adecuado (por ejemplo, modo 'P' para paleta de colores)
img = img.convert('P')

# Obtén las dimensiones
width, height = img.size

# Extrae los datos de píxeles
pixels = list(img.getdata())

# Escribe los datos en un formato compatible con ensamblador
with open('datos_imagen.txt', 'w') as f:
    for i, pixel in enumerate(pixels):
        if i % width == 0:
            f.write('\n')
        f.write(f'{pixel}, ')