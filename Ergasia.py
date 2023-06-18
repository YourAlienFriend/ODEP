#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:14:01 2023

@author: manolis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA

data = pd.read_csv('./telco_2023.csv')


# scaler = MinMaxScaler()
# data_normalised = scaler.fit_transform(data)
#scaler = StandardScaler()
#data_standarised= scaler.fit_transform(data)

data = pd.read_csv('./telco_2023.csv')
df = pd.DataFrame(data, columns=data.columns)  

X = df.drop(['region', 'marital', 'gender', 'callid', 'callwait', 'custcat', 'churn'], axis=1)
choice=1
colors= ["red","green","blue","magenta","black","orange","pink"]
while (choice>=1) and (choice<=3):
    try:
        choice=int(input("Give clustering method : \n 1.kmeans \n 2.agglomerative \n 3.DBscan \n Input:"))
        break
    except ValueError:
        print("Wrong Iput only Integers (1-3)")
        
        
if choice==1 :
    #elbow
    model = KMeans(random_state=42,n_init='auto')
    visualizer = KElbowVisualizer(model, k=(1,30))  
    visualizer.fit(X)
    visualizer.show()
    
   
    
    
    # επιλογή του elbow
    k = visualizer.elbow_value_
    final_model = KMeans(n_clusters=k, random_state=42)
    final_model.fit(X)
    
    # Obtain the cluster labels for the data points
    cluster_labels = final_model.labels_
    kmeansdata=df
    kmeansdata['cluster']=cluster_labels+1
    num_unique_values = len(kmeansdata['cluster'])


    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(kmeansdata)
    churn_rate = kmeansdata.groupby('cluster')['churn'].mean()
    problematic_cluster = churn_rate.idxmax()
   
elif choice==2:
    Z = linkage(X, method='complete', metric='euclidean')

# Plot the dendrogram
    plt.figure(figsize=(70,20))
    dn = dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

    
    
    unique_colors=set(dn['color_list'])
    k=len(unique_colors)-1
    
    
    agglomerative_model = AgglomerativeClustering(n_clusters=k)
    cluster_labels = agglomerative_model.fit_predict(X)
    agnesdata=df
    agnesdata['cluster']=cluster_labels
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(agnesdata)
    
    churn_rate = agnesdata.groupby('cluster')['churn'].mean()
    problematic_cluster = churn_rate.idxmax()
   

elif choice==3:
    k = 2 # Ορίζω την τιμή του k για το γράφημα της k-απόστασης
    neigh = NearestNeighbors(n_neighbors=k+1) # Λαμβάνω υπόψη k+1 γείτονες για να συμπεριλάβει και το ίδιο το σημείο 
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    distances = sorted(distances[:, k], reverse=False) # Ταξινομώ τις αποστάσεις σε αύξουσα σειρά
    sorted_indices = np.argsort(distances) # Ταξινομώ τα indexes βάσει των ταξινομημένων αποστάσεων
    sorted_distances = np.array(distances)[sorted_indices] # Ταξινομώ τις αποστάσεις βάσει των ταξινομημένων δεικτών
    
    
    plt.plot(sorted_indices, sorted_distances)
    plt.xlabel('Data Points')
    plt.ylabel('k-Distance')
    plt.title('k-Distance Graph')
    plt.show() 
    
    
    eps =15
    
    
    dbscan_model = DBSCAN(eps=eps,min_samples=12)
    cluster_labels = dbscan_model.fit_predict(X)
    
    dbscandata=df
    dbscandata['cluster']=cluster_labels
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dbscandata)
    churn_rate = dbscandata.groupby('cluster')['churn'].mean()
    problematic_cluster = churn_rate.idxmax()
   

#οπτικοποίηση αποτελεσμάτων
unique_labels = np.unique(cluster_labels)
for label in unique_labels:
    mask = (cluster_labels == label)
    plt.scatter(principal_components[mask, 0], principal_components[mask, 1], c=colors[label], label=f'Cluster {label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot with Clusters')
plt.legend()
plt.show()

#Εκτύωση προβληματικού cluster
print("Problematic cluster is ", problematic_cluster)
print(churn_rate)

#count = dbscandata[dbscandata['cluster'] == 1]['churn'].value_counts()[1]


