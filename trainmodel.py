import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import os
import pandas as pd
import glob 
import scipy.io.wavfile
import sys
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import model_from_json


def process():
	mylist= os.listdir('RawData/')
	#[0-neutral,1-calm,2-happy,3-sad,4-angry,5-fearful,6-disgust,7-surprised]
	feeling_list=[]
	for item in mylist:
	    if item[6:-16]=='01':
	        feeling_list.append(0)
	    elif item[6:-16]=='02':
	        feeling_list.append(1)
	    elif item[6:-16]=='03':
	        feeling_list.append(2)
	    elif item[6:-16]=='04':
	        feeling_list.append(3)
	    elif item[6:-16]=='05':
	        feeling_list.append(4)
	    elif item[6:-16]=='06':
	        feeling_list.append(5)
	    elif item[6:-16]=='07':
	        feeling_list.append(6)
	    elif item[6:-16]=='08':
	        feeling_list.append(7)

	labels = pd.DataFrame(feeling_list)



	df = pd.DataFrame(columns=['feature'])
	bookmark=0
	for index,y in enumerate(mylist):
	    X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
	    sample_rate = np.array(sample_rate)
	    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=13),axis=0)
	    feature = mfccs
	    df.loc[bookmark] = [feature]
	    bookmark=bookmark+1    
	print(df[:5])


	df3 = pd.DataFrame(df['feature'].values.tolist())
	newdf = pd.concat([df3,labels], axis=1)
	#rnewdf = newdf.rename(index=str, columns={"0": "label"})
	rnewdf = shuffle(newdf)
	print("shuffle",(newdf[:10]))
	rnewdf=newdf.fillna(0)

	print(rnewdf)
	X = rnewdf.iloc[:, :-1]
	y = rnewdf.iloc[:, -1:]
	y = to_categorical(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

	print(X_train)
	print(y_train)
	x_traincnn =np.expand_dims(X_train, axis=2)
	x_testcnn= np.expand_dims(X_test, axis=2)
	
	model = Sequential()
	model.add(Conv1D(256, 5,padding='same',input_shape=(216,1)))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 5,padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling1D(pool_size=(8)))
	model.add(Conv1D(128, 5,padding='same',))
	model.add(Activation('relu'))
	model.add(Conv1D(128, 5,padding='same',))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(8))
	model.add(Activation('softmax'))
	opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
	cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=50, validation_data=(x_testcnn, y_test))
	
	plt.plot(cnnhistory.history['loss'])
	plt.plot(cnnhistory.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.pause(5)
	plt.show(block=False)
	plt.close()	

	model_name = 'Emotion_Voice_Detection_Model.h5'
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	# Save model and weights
	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	preds = loaded_model.predict(x_testcnn,batch_size=32,verbose=1)
	preds1=preds.argmax(axis=1)
	abc = preds1.astype(int).flatten()
	print(abc)
	#predictions = (lb.inverse_transform((abc)))
	preddf = pd.DataFrame({'predictedvalues': abc})
	actual=y_test.argmax(axis=1)
	abc123 = actual.astype(int).flatten()
	#actualvalues = (lb.inverse_transform((abc123)))
	print(abc123)
	actualdf = pd.DataFrame({'actualvalues': abc123})
	finaldf = actualdf.join(preddf)
	finaldf.groupby('actualvalues').count()
	finaldf.groupby('predictedvalues').count()
	finaldf.to_csv('Predictions.csv', index=False)
