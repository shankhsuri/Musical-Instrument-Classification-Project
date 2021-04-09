#General
import numpy as np
import itertools
import pandas as pd

# System
import os, fnmatch
import pickle
import warnings
# Visualization
import seaborn #visualization library, must be imported before all other plotting libraries
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Random Seed
from numpy.random import seed
seed(1)

# Audio
import librosa.display, librosa
import streamlit as st

from urllib.request import urlopen
from sklearn.externals import joblib


modelknn = pickle.load(open('models/knnmodel.pkl','rb'))
modellabel=pickle.load(open('models/labelencoder.pkl','rb'))
modelscaler=pickle.load(open('models/scaler.pkl','rb'))

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

# Signal Processing Parameters
fs = 44100         # Sampling Frequency
n_fft = 2048       # length of the FFT window
hop_length = 512   # Number of samples between successive frames
n_mels = 128       # Number of Mel bands
n_mfcc = 13        # Number of MFCCs



def get_features(y, sr=fs):
    S = librosa.feature.melspectrogram(y, sr=fs, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    feature_vector = np.mean(mfcc,1)
    #feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
    return feature_vector


            
            
def main():
  menu = ["Tool","More"]
  
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Tool':
    st.header("Music Instrument Classifier")
    st.write("A machine learning tool which can detect the instrument class which the audio file belongs to")
    
    uploaded_file = st.file_uploader('Upload File',type='wav')
    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=fs)
        y/=y.max() #Normalize
        feat = get_features(y, sr)
        features=modelscaler.transform(feat.reshape(1,-1))
        

    
    if st.button('Generate Result'):
      pred=modelknn.predict(features)
      st.write("The uploaded audio file is from the",modellabel.inverse_transform(pred)[0],"instrument class")
            
      
if __name__ == '__main__':
	main()
 
