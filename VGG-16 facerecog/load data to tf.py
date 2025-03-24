import tensorflow as tf #TensorFlow est une bibliothèque open source développée par Google principalement pour les applications 
#d'apprentissage profond.TensorFlow accepte les données sous la forme de tableaux multidimensionnels de dimensions supérieures 
#appelés tenseurs. Les tableaux multidimensionnels sont très pratiques pour gérer de grandes quantités de données.
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img
#Charger des images augmentées dans l'ensemble de données Tensorflow
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False) #TensorFlow Datasets est une collection  
#d'ensemble de données prêts à être utilisés avec TensorFlow
train_images = train_images.map(load_image) #images d'entrainement
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)
test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False) #images de test
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)
val_images = tf.data.Dataset.list_files('aug_data\\validation\\images\\*.jpg', shuffle=False) #images de validation
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)
train_images.as_numpy_iterator().next()
#Créer une fonction de chargement d'étiquettes
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']
#Charger les étiquettes dans l'ensemble de données Tensorflow
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
val_labels = tf.data.Dataset.list_files('aug_data\\validation\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
train_labels.as_numpy_iterator().next()
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))
#créer bases de données finales(images/labels)
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)
data_samples = train.as_numpy_iterator()

#Face Recognition deep CNN
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
vgg = VGG16(include_top=False) #telecharger VGG-16
vgg.summary()
#créer notre CNN
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)

    # Modèle de classification 
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Modèle de boîte englobante
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker
facetracker = build_model()
facetracker.summary()
X, y = train.as_numpy_iterator().next()
#Définir les pertes et les optimiseurs
#definir l'optimiseur et le pas
batches_per_epoch = len(train) #nb de lots par epoque(Une époque signifie que chaque échantillon de l'ensemble de données d'apprentissage 
#a eu l'occasion de mettre à jour les paramètres du modèle interne)
lr_decay = (1./0.75 -1)/batches_per_epoch #reduction lente du pasjusqu'à l'obtention du minimum
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)#Adam est un algorithme d'optimisation qui peut être utilisé 
#à la place de la procédure classique de descente de gradient stochastique pour mettre à jour les poids du réseau de manière itérative 
#en fonction des données d'apprentissage.

#Créer une erreur de localisation et une erreur de classification
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 
    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    return delta_coord + delta_size
classloss = tf.keras.losses.BinaryCrossentropy() #erreur cross entropy
regressloss = localization_loss
#entrainer notre réseau de neurones
#Créer une classe de modèle personnalisée
class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        X, y = batch #lot de data
        with tf.GradientTape() as tape: #fct de clacul dans notre réseau
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)#calculer le gradient
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))# appliquer la descente du gradient
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): #calcul des erreurs pour les images de test
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)#compiler le modele
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback]) #adapter le modèle et entrainement
print(hist.history)
#faire des prédictions sur les images de test
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): #affiche 4 images
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    
    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)#trace un rectangle sur le visage s'il ya reconnaissance
    
    ax[idx].imshow(sample_image)    
from keras.models import load_model
facetracker.save('facetracker.h5')#sauvegarder notre modele entrainé
  