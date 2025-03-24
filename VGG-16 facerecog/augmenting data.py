import albumentations as alb
import os
import json
import numpy as np
import cv2
#Appliquer une augmentation d'image sur des images et des étiquettes à l'aide d'albumentations
#Configurer le pipeline de transformation des albumentations
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
#Exécuter le pipeline d'augmentation
for partition in ['train','test','validation']: 
    for image in os.listdir(os.path.join('data', partition, 'images')):
        #Charger une image de test et une annotation avec OpenCV et JSON
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            #Extraire les coordonnées et redimensionner pour correspondre à la résolution de l'image
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))

        try: 
            for x in range(70):#répeter 70fois pour chaque image ce qui nous donne 90x70 images
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])#appliquer augmentations
                #on envoie ces images augmentées dans un dossier 'aug-data' contenat 3 dossiers 'test','train' et 'validation'
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
                #etiquettes d'images augmentées
                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(e)
        