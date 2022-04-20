import numpy as np                                
import pandas as pd                               
import seaborn as sns
import matplotlib.pyplot as plt                   
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import time
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
#importo il dataset
dataset= pd.read_csv('beer.csv')
dataset.pop('Brew_No')#rimuovo l'identificativo
#trasformo le categorie di birre da stringhe ad interi 
dataset['style'] = dataset['style'].map( {'Premium Lager': 0, 'IPA': 1,'Light Lager': 2}).astype(int)
#vado a scalare i dati
scaler = StandardScaler()
scaled_array = scaler.fit_transform(dataset)
scaled_dataset = pd.DataFrame( scaled_array, columns = dataset.columns )
print(scaled_dataset.describe())
print(scaled_dataset)






