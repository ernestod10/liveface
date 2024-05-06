import cv2
import numpy as np
import os 

# Cargar el clasificador de rostros pre-entrenado Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo pre-entrenado de detección de género
if os.path.isfile('gender_deploy.prototxt') and os.path.isfile('gender_net.caffemodel'):
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    gendercheck = True
else:
    print("Error: No se pudo cargar el modelo de detección de genero.")
    gendercheck = False

# Abrir la cámara web
cap = cv2.VideoCapture(0)  # Usa el índice apropiado para tu cámara web

while True:
    # Capturar video desde la cámara web
    ret, frame = cap.read()

    # Verificar si el fotograma es válido
    if not ret:
        print("Error: No se pudo capturar el fotograma desde la cámara web.")
        break

    # Convertir el fotograma a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Inicializar una bandera para indicar si se detecta un rostro real
    real_face_detected = False
    

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in faces:
        # Extraer la región del rostro
        face_roi = gray[y:y+h, x:x+w]

        # Calcular la varianza de textura de la región del rostro
        texture_variance = np.var(face_roi)

        # Establecer un umbral para diferenciar entre un rostro real y una foto
        threshold = 2800


        # Si la varianza de textura está por encima del umbral, considerarlo un rostro real
        if texture_variance > threshold:
            # Calcular la confianza de que sea un rostro real
            real_face_confidence = (texture_variance - threshold) / (255 - threshold)
            real_face_confidence_percentage = 100 + round(real_face_confidence * 100, 2)
            print(f"Confianza de que sea un rostro real: {real_face_confidence_percentage}%")
            # Realizar la detección de género
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Convertir la imagen en escala de grises a RGB
            blob = cv2.dnn.blobFromImage(face_roi_rgb, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            if gendercheck:
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = "Masculino" if gender_preds[0][0] > gender_preds[0][1] else "Femenino"
                print(f"Se detecta un rostro {gender}.")
                cv2.putText(frame, f"Genero: {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("Se detecta un rostro real.")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print("No se detecta un rostro real.")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Dibujar un rectángulo rojo alrededor del rostro

    # Mostrar el fotograma
    cv2.imshow("Detección de Rostros Reales", frame)

    # Romper el bucle cuando se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara web y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
