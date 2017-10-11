
import numpy as np
import tflearn
import tensorflow as tf
tf.reset_default_graph()

# Traitement des données
def preprocess(liste,colonneaenlever):
    for x in liste:
        x.pop(colonneaenlever) 
    for x in liste:
        x[0]=float(x[0])
        x[1]=float(x[1])
        if x[2]=='femme':
            x[2]=1.
        else:
            x[2]=0.
    return np.array(liste,dtype=np.float32)

# On définit l'indice des colonnes à ignorer
colonneaenlever=0

# On importe la BDD
from tflearn.data_utils import load_csv
data, labels = load_csv('./BDD_exemple.csv', target_column=4,categorical_labels=True, n_classes=2)
print("label:",labels)
print(data)
data = preprocess(data, colonneaenlever)
print(data)

#On créée le réseau de neuronne
net = tflearn.input_data(shape=[None,3])
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 2, activation='sigmoid')
net = tflearn.regression(net)

#On définit le type d'entrainement du modèle
model = tflearn.DNN(net)

#On commence à nourir le réseau de neuronne
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
#



#On prend un exemple
exemple = [['Coby','20','15','femme']]
# On le traite comme les données
exemple=preprocess(exemple,colonneaenlever)
# Prédiction du résultat
pred = model.predict(exemple)
print(" Cobaye :", pred[0][1])



