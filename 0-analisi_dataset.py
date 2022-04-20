import numpy as np                                
import pandas as pd                               
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn import datasets                     
import time
from sklearn.manifold import TSNE

#importo il dataset
data= pd.read_csv('beer.csv')
print(data)
#mi stampo una descrizione generale delle varie feature
print(data.describe().T)
#codifico con ordinal encoding
data['style'] = data['style'].map( {'Premium Lager': 0, 'IPA': 1,'Light Lager': 2}).astype(int)

#frequenza delle birre per feature
for feature in ['Brew_No','OG','ABV','pH','IBU','style']:

    plt.hist(data.loc[~data[feature].isnull()][feature])
    plt.title(feature)
    plt.show()
#tolgo la colonna brew number
data.pop('Brew_No')
#stampo la tabella di correlazione
print(data.corr())
#stampo i vari grafici delle feature
sns.pairplot(data)
plt.show()
#vado a stampare la heatmap
plt.figure(figsize = (15,6))
sns.heatmap( data.corr(), annot=True)
plt.show()

#tsne
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=1000)
tsne_results = tsne.fit_transform(data)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data['tsne-2d-one'] = tsne_results[:,0]
data['tsne-2d-two'] = tsne_results[:,1]


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    
    palette=sns.color_palette("hls", 10),
    data=data,
    legend="full",
    alpha=0.3
)
plt.show()
