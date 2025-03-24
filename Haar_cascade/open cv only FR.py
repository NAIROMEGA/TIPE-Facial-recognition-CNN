import numpy as np
import cv2
import pickle
face_cascade=cv2.CascadeClassifier('CASCADE\data\haarcascade_frontalface_alt.xml') #Créer notre classificateur de faces frontales
recognizer = cv2.face.LBPHFaceRecognizer_create() #création de notre reconnaiseur
recognizer.read("face-trainner.yml") #charger notre reconnaisseur entrainé

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f) #charger nos etiquettes du fchier pickle
	labels = {v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0) # pour la capture avec la webcam en temps réel
while True:
    ret, frame=cap.read() # Lire la première image de la vidéo
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertir l'image en niveaux de gris
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)# Utiliser le classificateur pour détecter les visages  

    for (x,y,w,h)in faces:
     roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
     roi_color = frame[y:y+h, x:x+w]
     id_, conf = recognizer.predict(roi_gray)
     if conf>=4 and conf <= 85: #conditions de reconnaisance
      font = cv2.FONT_HERSHEY_SIMPLEX
      name = labels[id_]
      color = (255, 255, 255)
      stroke = 2
      cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA) #écrire le nom de la personne identifiée
     img_item="my-image.png"
     cv2.imwrite(img_item,roi_gray)
     color=(255,0,0) #BGR
     stroke=2
     width=x+w
     height=y+h
     cv2.rectangle(frame,(x,y),(width,height),color,stroke) # Dessiner des rectangles autour des visages détectées
    cv2.imshow('frame', frame) # Afficher l'image avec les visages détectées
    if cv2.waitKey(20) & 0xFF == ord('q'): # appuyez sur 'q' sur le clavier pour quitter
     break
# Libérer les ressources et fermer les fenêtres d'affichage
cap.release() 
cv2.destroyAllWindows()