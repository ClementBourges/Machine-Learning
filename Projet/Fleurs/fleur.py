import tflearn
from tflearn.data_utils import *

path = "fleurs.txt"
maxlen = 25
dictionnaire=None 

#On créée le dictionnaire à partir de la base de données
X, Y, dictionnaire = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                         pre_defined_char_idx=dictionnaire)

#On créée le réseau de neurone
g = tflearn.input_data([None, maxlen, len(dictionnaire)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(dictionnaire), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=dictionnaire,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_fleurs')


#On entraîne le réseau
for i in range(14):
    m.fit(X, Y, validation_set=0.1, batch_size=128,
    n_epoch=1, run_id='fleurs')

#On génère une séquence
graine =random_sequence_from_textfile(path, maxlen)    
print("TEST:")
print("Température =1 :")
print(m.generate(300, temperature=1, seq_seed=graine))
print("\n\nTempérature =0.5 :")
print(m.generate(300, temperature=0.5, seq_seed=graine))
    

