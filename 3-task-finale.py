import numpy as np                                
import pandas as pd                               
import seaborn as sns
import matplotlib.pyplot as plt                   
from sklearn import datasets                     
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans              
from sklearn import metrics
import time
from sklearn.manifold import TSNE

print("1:birra moretti")
print("2:HB waisse")
print("3:tennent's")
print("4:brewdog ipa")
print("5:coors light")
print("6:peroni leggera")
scelta=input("scegli una birra che ti piace")




#importo il dataset
final_dataset= pd.read_csv('beer.csv')
dataset= pd.read_csv('beer.csv')
#rimuovo la colonna brew n
dataset.pop('Brew_No')
#faccio encoding 
dataset['style'] = dataset['style'].map( {'Premium Lager': 0, 'IPA': 1,'Light Lager': 2}).astype(int)
#applico lo z-score
scaler = StandardScaler()
scaled_array = scaler.fit_transform(dataset)
scaled_dataset = pd.DataFrame( scaled_array, columns = dataset.columns )
print(scaled_dataset.describe())
print(scaled_dataset)

#-------------------
#clusterizzo con il k means
kmeans_model = KMeans(n_clusters = 3)
kmeans_model.fit(scaled_dataset)
centroids = kmeans_model.cluster_centers_
#print(kmeans_model.cluster_centers_.shape)
#print(kmeans_model.labels_)
scaled_dataset["cluster"] = kmeans_model.labels_
final_dataset["cluster"] = kmeans_model.labels_
#vado a plottare i risultati del k-means
sns.pairplot(data = scaled_dataset, hue = "cluster", palette = "crest")
plt.show()
#--------------------------
#cerco l'indice di Silhouette per verificare il numero ottimale di cluster
k_to_test = range(2,25,1) 
silhouette_scores = {}

for k in k_to_test:
    model_kmeans_k = KMeans( n_clusters = k )
    model_kmeans_k.fit(scaled_dataset.drop("cluster", axis = 1))
    
    labels_k = model_kmeans_k.labels_
    score_k = metrics.silhouette_score(scaled_dataset.drop("cluster", axis=1), labels_k)
    silhouette_scores[k] = score_k
    print("Tested kMeans with k = %d\tSS: %5.4f" % (k, score_k))
    
print("Done!")

#plotto il grafico con il numero ottimale di cluster
plt.figure(figsize = (16,5))
plt.plot(silhouette_scores.values())
plt.xticks(range(0,23,1), silhouette_scores.keys())
plt.title("Silhouette Metric")
plt.xlabel("k")
plt.ylabel("Silhouette")
#plt.axvline(1, color = "r")
plt.show()

print(final_dataset)
#inserisco il dataset clusterizzato in un file csv
final_dataset.to_csv("fin_dat.csv")
#gestisco la scelta fatta inizialmente
if(scelta=="1"):
    #vado a eseguire delle query per estrappolarmi le birre da cosigliare
    clus=final_dataset.query("Brew_No == -1")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare) 
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)

elif(scelta=="2"):
    clus=final_dataset.query("Brew_No == -2")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare)
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)
    
elif(scelta=="3"):
    clus=final_dataset.query("Brew_No == -3")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare)
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)
elif(scelta=="4"):
    clus=final_dataset.query("Brew_No == -4")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare)
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)
elif(scelta=="5"):
    clus=final_dataset.query("Brew_No == -5")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare)
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)
elif(scelta=="6"):
    clus=final_dataset.query("Brew_No == -6")
    cluster_da_usare=clus['cluster'].astype(int)
    print(cluster_da_usare)
    scelta=input("inserisci il cluster da utilizzare")
    consigli=final_dataset.query("cluster == "+scelta+"")
    print(consigli)
    
