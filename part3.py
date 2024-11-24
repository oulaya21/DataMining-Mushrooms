import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import confusion_matrix
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE

# Load the dataset
data_path = 'C:/Users/hp/Desktop/DataMining/mushroom/agaricus-lepiota.data'
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv(data_path, header=None, names=columns)

# Remove the 'class' attribute and perform one-hot encoding
data_for_clustering = pd.get_dummies(data.drop(['class'], axis=1))

# Perform hierarchical clustering
cah = AgglomerativeClustering(n_clusters=7)
cah_clusters = cah.fit_predict(data_for_clustering)

# Limit the number of samples for the dendrogram to avoid clutter
max_samples = 200
if len(data_for_clustering) > max_samples:
    data_for_dendrogram = data_for_clustering.sample(n=max_samples, random_state=42)
else:
    data_for_dendrogram = data_for_clustering

# Plot dendrogram for CAH
plt.figure(figsize=(15, 10))
plt.title('Dendrogram - CAH')
dendrogram = sch.dendrogram(sch.linkage(data_for_dendrogram, method='ward'), leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Perform k-means clustering
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans_clusters = kmeans.fit_predict(data_for_clustering)

# Add the cluster labels to the dataset
data['Cluster'] = kmeans_clusters

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(data_for_clustering)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans_clusters, palette='viridis', legend='full')
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Cluster')
plt.show()

# Compare the clusters with the 'class' attribute using confusion matrix
true_labels = data['class'].astype(str).values
cah_clusters_str = cah_clusters.astype(str)
kmeans_clusters_str = kmeans_clusters.astype(str)

cah_cm = confusion_matrix(true_labels, cah_clusters_str)
kmeans_cm = confusion_matrix(true_labels, kmeans_clusters_str)

# Print confusion matrices
print('Confusion Matrix - CAH:')
print(cah_cm)
print('\nConfusion Matrix - K-means:')
print(kmeans_cm)
