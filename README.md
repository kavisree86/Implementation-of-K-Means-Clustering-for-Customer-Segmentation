# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Pick customer segment quantity (k)

2.Seed cluster centers with random data points.

3.Assign customers to closest centers.

4.Re-center clusters and repeat until stable. 


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: kavisree.s
RegisterNumber:  212222047001

import pandas as pd   
import numpy as np  
from sklearn.cluster import KMeans   
from sklearn.metrics.pairwise import euclidean_distances   
import matplotlib.pyplot as plt   
data=pd.read_csv('/content/Mall_Customers_EX8.csv')   
data   
X=data[['Annual Income (k$)','Spending Score (1-100)']]   
X  
plt.figure(figsize=(4,4))   
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])   
plt.xlabel('Annual Income (k$)')   
plt.ylabel('Spending Score (1-100)')   
plt.show()   
K=5   
Kmeans=KMeans(n_clusters=K)   
Kmeans.fit(X)   
centroids = Kmeans.cluster_centers_   
labels = Kmeans.labels_   
print("Centriods:")   
print(centroids)   
print("Labels:")     
print(labels)   
colors=['r','g','b','c','m']   
for i in range(K):   
  cluster_points = X[labels == i]   
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster {i+1}')   
  distances=euclidean_distances(cluster_points,[centroids[i]])   
  radius=np.max(distances)   
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)  
  plt.gca().add_patch(circle)   
  plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')   
plt.title('K-means Clustering')   
plt.xlabel('Annual Income (k$)')   
plt.ylabel('Spending Score (1-100)')  
plt.legend()   
plt.grid(True)   
plt.axis('equal')   
plt.show()   
*/
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)


<img width="509" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/c4e06017-920f-4694-a127-8d37e3be0fa1">

<img width="397" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/a8961651-9f20-4313-859c-0857a84ed7f3">

<img width="374" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/4abee026-0cf5-465b-8a21-2e476d023589">

<img width="492" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/57b51845-e755-496d-9e7e-53e0a16a7ee1">

<img width="472" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/ce50307f-cca3-4dd8-97b3-1653f5d60df7">
<img width="470" alt="image" src="https://github.com/kavisree86/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145759687/02127d08-82a4-4337-a519-f8ff31329e56">






## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
