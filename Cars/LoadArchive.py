import scipy.io

# Carga del archivo .mat
data = scipy.io.loadmat('cars_annos.mat')

# Explorar las claves del archivo
print(data.keys())

# Visualizar información específica (ejemplo: nombres de clases)
class_names = data['class_names']  # Cambia 'class_names' si la clave es diferente
for idx, name in enumerate(class_names[0]):
    print(f"Clase {idx + 1}: {name[0]}")
