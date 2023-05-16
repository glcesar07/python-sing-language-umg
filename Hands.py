import cv2
import mediapipe as mp
import os

#----------------------------- Creamos la carpeta donde almacenaremos el entrenamiento ---------------------------------
nombre = 'Lunes'
direccion = 'C:/www/UMG/IA/pythonSingLanguage/Fotos/Validacion'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ',carpeta)
    os.makedirs(carpeta)

#Asignamos un contador para el nombre de la fotos
cont = 0

#Leemos la camara
cap = cv2.VideoCapture(2)

#----------------------------Creamos un obejto que va almacenar la deteccion y el seguimiento de las manos------------
clase_manos  =  mp.solutions.hands
manos = clase_manos.Hands() #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------------Metodo para dibujar las manos---------------------------
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos criticos de la mano


while (1):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[0]  # Punto central
                desplazamiento_dedos = 150  # Ajusta este valor para dar más margen en los dedos
                desplazamiento_palma = 150  # Ajusta este valor para dar más margen en la palma
                x1, y1 = (pto_i5[1] - desplazamiento_dedos), (pto_i5[2] - desplazamiento_dedos - desplazamiento_palma)
                ancho, alto = (x1 + desplazamiento_dedos), (y1 + desplazamiento_dedos + desplazamiento_palma)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            #dedos_reg = cv2.resize(dedos_reg,(400,400), interpolation = cv2.INTER_CUBIC) #Redimensionamos las fotos
            #cv2.imwrite(carpeta + "/Mano_{}.jpg".format(cont),dedos_reg)
            #cont = cont + 1





    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break
cap.release()
cv2.destroyAllWindows()
