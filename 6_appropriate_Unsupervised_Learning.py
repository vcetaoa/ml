import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Load the dataset
data = pd.read_csv('6customers.csv')

# Step 2: Data Preprocessing
# Encode categorical variables (CHANNEL and REGION columns)
data['Region'] = data['Region'].map({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})
data['Channel'] = data['Channel'].map({1: 'Hotel', 2: 'Retailer'})

# Step 3: Feature Selection
# Drop the 'Channel' and 'Region' columns for clustering
X = data.drop(columns=['Channel', 'Region'])

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply K-Means Clustering
# Determine the optimal number of clusters using the Elbow Method
wcss = [] # List to store WCSS values for different k
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Elbow Method: Plotting WCSS to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.show()

# Choose the optimal number of clusters based on the elbow point (example: 3 clusters)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualizing the clusters using PCA (Principal Component Analysis)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter Plot: Visualizing customer segments in 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments (Clustering Results)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Cluster Centers: Visualizing the centers in 2D
centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers) # Project the cluster centers onto PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
plt.title('Clusters and Their Centers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Step 7: Cluster Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=data, palette='viridis')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

# Step 8: Pairplot to visualize original feature relationships
sns.pairplot(data.drop(columns=['Channel', 'Region']), hue='Cluster', palette='viridis')
plt.show()

# Optional: Save the clustered data to a new CSV file
data.to_csv('clustered_wholesale_customers.csv', index=False)
