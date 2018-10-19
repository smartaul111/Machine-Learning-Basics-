#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("THe elbow method")
plt.xlabel("no of cluster")
plt.ylabel("WCSS")
plt.show()

#apply kmeans

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0 )
y_kmeans=kmeans.fit_predict(X)

#visualization

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label="carefull")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label="standard")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='gold',label="target")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label="careless")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label="sensible")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='green',label="centroids")
plt.title("cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score(1-100)")
plt.legend()
plt.show()