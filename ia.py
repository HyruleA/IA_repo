import numpy as np
import cv2 as cv

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
i = 0

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in rostros:
        # Recortar la región del rostro detectado (100x100)
        frame2 = frame[y:y+h, x:x+w]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
        
        # Recortar una segunda región del rostro (80x80)
        frame3 = frame[y:y+h, x:x+w]
        frame3 = cv.resize(frame3, (80, 80), interpolation=cv.INTER_AREA)
        
        # Convertir ambos recortes a escala de grises
        gray_100x100 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        gray_80x80 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
        
        # Aplicar umbral para convertir en imagen binaria (blanco y negro)
        _, binary_100x100 = cv.threshold(gray_100x100, 127, 255, cv.THRESH_BINARY)
        _, binary_80x80 = cv.threshold(gray_80x80, 127, 255, cv.THRESH_BINARY)

        # Guardar las imágenes recortadas y binarizadas
        cv.imwrite('/home/likcos/recorte/100x100_' + str(i) + '.jpg', binary_100x100)
        cv.imwrite('/home/likcos/recorte/80x80_' + str(i) + '.jpg', binary_80x80)
        
        # Mostrar las imágenes en binario
        cv.imshow('100x100 rostro binario', binary_100x100)
        cv.imshow('80x80 rostro binario', binary_80x80)
        
        i += 1

    # Mostrar el frame original con los rostros detectados
    cv.imshow('Rostros', frame)
    
    # Salir del bucle si se presiona la tecla 'Esc' (código 27)
    if cv.waitKey(1) == 27:
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv.destroyAllWindows()