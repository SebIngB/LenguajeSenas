import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from keras.models import load_model
import serial

ard = serial.Serial('COM7',9600)

model = load_model('Model1.h5')

model.summary()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 200

labels = ['A','E','I','O','U']

label_data = {0:(b'0', 'LED RED ENCENDIDO'),
              1:(b'1', 'LED GREEN ENCENDIDO'),
              2:(b'2', 'LED BLUE ENCENDIDO'),
              3:(b'3', 'LED RED Y GREEN ENCENDIDO'),
              4:(b'4', 'LED RED, GREEN Y BLUE ENCENDIDO')}

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 # Defino un tamaño de la imagen en blanco
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        imgCropShape = imgCrop.shape #Obtengo las dimensiones de la deteccion de la mano
   
        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h #constante
            wCal = math.ceil(k*w)  # calculo weight
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
    
        else:
            k = imgSize / w  #constante
            hCal = math.ceil(k * h)  # calculo weight
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal + hGap:, :] = imgResize

        imagen_redimensionada = np.expand_dims(imgWhite, axis=0)
        x_data = np.array(imagen_redimensionada)

        prediction = model.predict(x_data, verbose=0)

        #print(np.argmax(prediction))
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255,0,255),cv2.FILLED)
        cv2.putText(imgOutput, labels[np.argmax(prediction)], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 2,(255,255,255),2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255,0,255),4)

        prediction = np.argmax(prediction)
        if prediction in label_data:
            datos_enviar, mensaje = label_data[prediction]
            ard.write(datos_enviar)
            print(mensaje)
        
        #cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    else:
        ard.write(b'5')
        datos = ard.readline()
        print(datos)


    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("e"):

        cap.release()
        cv2.destroyAllWindows()