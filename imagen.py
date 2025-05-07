
import cv2
import matplotlib.pyplot as plt
import easyocr

placa=['placa_q.jpg',
       'placa_2.jpg',
       'placa_3.jpg',
       'placa_4.jpg']

reader = easyocr.Reader(['en'], gpu=False)

def procesar_imagenes(lista):

    #Leer la imagen
    img = cv2.imread(lista)

    #Convertir la imagen a escalas de grises
    filtro = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #Añadir un filtro bilateral para reducir el ruido, respetando los bordes
    bfiltro = cv2.bilateralFilter(filtro, 11, 17, 17)

    #Se aplica el algoritmo de deteccion Canny
    bordes = cv2.Canny(bfiltro, 30, 200)
    return img, bordes
cont=1
for lista in placa:
    img_original, bordes = procesar_imagenes(lista)

    # Mostrar imagen con bordes
    plt.imshow(bordes, cmap='gray')
    plt.title(f"Bordes de {lista}")
    plt.axis('off')
    plt.show()

    # Aplicar OCR sobre la imagen original
    resultado = reader.readtext(img_original)

    print(f"Texto de placa: {cont}")
    for _, texto, _ in resultado:
      print("-", texto)

    #Contador para detener el programa despues de leer las imagenes
    if cont==4: 
        break
    cont +=1
    
