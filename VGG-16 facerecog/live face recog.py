import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
facetracker = load_model('facetracker.h5') #charger le fichier contenant notre modèle entrainé 
cap = cv2.VideoCapture(0) #videocapture à travers la webcam
while cap.isOpened():
    _ , frame = cap.read()# Lire la première image de la vidéo
    frame = frame[50:500, 50:500,:]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir l'image en niveaux de gris
    resized = tf.image.resize(rgb, (120,120))
    yhat = facetracker.predict(np.expand_dims(resized/255,0))#prédiction
    sample_coords = yhat[1][0]
    if yhat[0] > 0.9: 
        # Contrôle le rectangle principal
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Contrôle le rectangle de l'étiquette
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Contrôle le texte rendu
        cv2.putText(frame, 'oussama', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame) # Afficher l'image avec le visage détecté
    if cv2.waitKey(1) & 0xFF == ord('q'):# appuyez sur 'q' sur le clavier pour quitter
        break
cap.release()
cv2.destroyAllWindows()# Libérer les ressources et fermer les fenêtres d'affichage