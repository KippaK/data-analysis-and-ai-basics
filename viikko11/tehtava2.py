import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Lataa data ja tutustu siihen
data = pd.read_csv('iris.csv')
print(data.head())

# Valitse halutut ominaisuudet klusterointia varten
X = data.iloc[:, [2, 3]].values

# Käytä Elbow Methodia löytääksesi optimaalisen klusterien määrän
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualisoi Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Käytä optimaalista klusterien määrää (tässä tapauksessa 3)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Muunna klusterit alkuperäisiksi lajikkeiksi
encoder = LabelEncoder()
data['species_encoded'] = encoder.fit_transform(data['species'])
species_map = {0: 'versicolor', 1: 'setosa', 2: 'virginica'}
data['cluster'] = [species_map[cluster] for cluster in y_kmeans]

# Tulosta ristiintaulukointi
conf_matrix = confusion_matrix(data['species'], data['cluster'], labels=['versicolor', 'setosa', 'virginica'])
print("Confusion Matrix:")
print(conf_matrix)

# Visualisoi klusterit
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of iris species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()
