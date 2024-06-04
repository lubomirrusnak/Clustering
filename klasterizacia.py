# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/lubom/python/jaspravim/Mall_Customers.csv')
#dataset = pd.read_csv('C:/Users/lubom/python/jaspravim/ulohy_student/kreditne karty/Wholesale customers data.csv')
#dataset = pd.read_csv('C:/Users/lubom/python/jaspravim/ulohy_student/5uloha/ObesityDataSet.csv')

rows_25_percent = int(0.25 * len(dataset))
rows_50_percent = int(0.50 * len(dataset))
rows_75_percent = int(0.75 * len(dataset))

#vzska vaha 
X = dataset.iloc[:, [2,3]].values
X_25 = dataset.iloc[:rows_25_percent, [2,3]].values
X_50 = dataset.iloc[:rows_50_percent, [2,3]].values
X_75 = dataset.iloc[:rows_75_percent, [2,3]].values

from sklearn.cluster import KMeans
wcss = []

wcss_25 = []
wcss_50 = []
wcss_75 = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
        # Pre 25% datasetu
    kmeans.fit(X_25)
    wcss_25.append(kmeans.inertia_)
    
        # Pre 50% datasetu
    kmeans.fit(X_50)
    wcss_50.append(kmeans.inertia_)
    
        # Pre 25% datasetu
    kmeans.fit(X_75)
    wcss_75.append(kmeans.inertia_)
    
    
# Vykreslenie grafu pre všetky veľkosti datasetu
#plt.figure(figsize=(12, 6))    
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCC')
plt.show()


plt.plot(range(1,11), wcss_25)
plt.title('The Elbow Method 25%')
plt.xlabel('Number of clusters')
plt.ylabel('WCC')
plt.show()


plt.plot(range(1,11), wcss_50)
plt.title('The Elbow Method 50%')
plt.xlabel('Number of clusters')
plt.ylabel('WCC')
plt.show()


plt.plot(range(1,11), wcss_75)
plt.title('The Elbow Method 75%')
plt.xlabel('Number of clusters')
plt.ylabel('WCC')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)#NEMENIT
kmeans25 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)#dobre
kmeans50 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)#dobre
kmeans75 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42) #DOBRE



y_kmeans = kmeans.fit_predict(X)
y_kmeans_25 = kmeans25.fit_predict(X_25)
y_kmeans_50 = kmeans50.fit_predict(X_50)
y_kmeans_75 = kmeans75.fit_predict(X_75)







# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Visualising the clusters
plt.scatter(X_25[y_kmeans_25 == 0, 0], X_25[y_kmeans_25 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_25[y_kmeans_25 == 1, 0], X_25[y_kmeans_25 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_25[y_kmeans_25 == 2, 0], X_25[y_kmeans_25 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X_25[y_kmeans_25 == 3, 0], X_25[y_kmeans_25 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X_25[y_kmeans_25 == 4, 0], X_25[y_kmeans_25 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans25.cluster_centers_[:, 0], kmeans25.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers 25%')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualising the clusters 50%
plt.scatter(X_50[y_kmeans_50 == 0, 0], X_50[y_kmeans_50 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_50[y_kmeans_50 == 1, 0], X_50[y_kmeans_50 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_50[y_kmeans_50 == 2, 0], X_50[y_kmeans_50 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_50[y_kmeans_50 == 3, 0], X_50[y_kmeans_50 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X_50[y_kmeans_50 == 4, 0], X_50[y_kmeans_50 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans50.cluster_centers_[:, 0], kmeans50.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers 50%')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualising the clusters
plt.scatter(X_75[y_kmeans_75 == 0, 0], X_75[y_kmeans_75 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_75[y_kmeans_75 == 1, 0], X_75[y_kmeans_75 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_75[y_kmeans_75 == 2, 0], X_75[y_kmeans_75 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_75[y_kmeans_75 == 3, 0], X_75[y_kmeans_75 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X_75[y_kmeans_75 == 4, 0], X_75[y_kmeans_75 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans75.cluster_centers_[:, 0], kmeans75.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers 75%')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()