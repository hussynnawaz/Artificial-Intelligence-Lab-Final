import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv('clusterdata_final.csv')
print(data.columns)
x_col = 'A'
y_col = 'B'
plt.figure(figsize=(8, 6))
plt.scatter(data[x_col], data[y_col])
plt.title('Scatter Chart')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.show()
def plot_clusters(data, n_clusters, x_col, y_col):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[[x_col, y_col]])   
    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        clustered_data = data[data['cluster'] == cluster]
        plt.scatter(clustered_data[x_col], clustered_data[y_col], label=f'Cluster {cluster}')
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
for n_clusters in [3, 4, 5]:
    plot_clusters(data.copy(), n_clusters, x_col, y_col)