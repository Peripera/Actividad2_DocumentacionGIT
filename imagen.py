
import cv2
import matplotlib.pyplot as plt
import easyocr

placa=['placa_q.jpg','placa_2.jpg','placa_3.jpg','placa_4.jpg']
reader = easyocr.Reader(['en'], gpu=True)

def procesar_imagenes(lista):
    img = cv2.imread(lista)#Leer la imagen
    filtro = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convertir la imagen a escalas de grises
    bfiltro = cv2.bilateralFilter(filtro, 11, 17, 17)#Añadir un filtro bilateral para reducir el ruido, respetando los bordes
    bordes = cv2.Canny(bfiltro, 30, 200)#Se aplica el algoritmo de deteccion Canny
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

    print(f"\Texto de placa: {cont}:")

    for _, texto, _ in resultado:
      print("-", texto)

