
import tensorflow as tf
import tflearn
import sys
import numpy as np
from tflearn.data_utils import *
tf.reset_default_graph()


maxlen=3
char_idx=None
path="huitlettre.txt"
layer_size=128
checkpoint_path='model'


#On créée le dictionnaire
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                         pre_defined_char_idx=char_idx)


#On créée le réseau de neurone
g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, layer_size, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, layer_size,return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, layer_size)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g,
                       optimizer='adam',
                       loss='categorical_crossentropy',
                       learning_rate=0.001
)

m = tflearn.SequenceGenerator(g,
                          dictionary=char_idx,
                          seq_maxlen=maxlen,
                          clip_gradients=5.0,
                          checkpoint_path=checkpoint_path)


#On entraîne le réseau
for i in range(100000):
    m.fit(X, Y, validation_set=0.1, batch_size=8,
    n_epoch=1, run_id='fleurs')


#On génère de nouvelles entrées
graine="sup"
taille=5
print("TEST:")
print("Température =1 :")
for k in range (7):
    print(m.generate(taille, temperature=1, seq_seed=graine))


print("\nTempérature =0.5 :")
print(m.generate(taille, temperature=0.5, seq_seed=graine))
for k in range (7):
    print(m.generate(taille, temperature=1, seq_seed=graine))



