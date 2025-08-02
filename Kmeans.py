import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# generate synthetic data for vehicles
np.random.seed(0)
data_size = 300
data = {
    'Weight': np.random.randint(1000, 3000, data_size),  
    'EngineSize': np.random.uniform(1.0, 4.0, data_size),  
    'Horsepower': np.random.randint(50, 300, data_size)  
}
df = pd.DataFrame(data)

# no labeels are needed for clustering
X = df

# perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Weight'], df['Horsepower'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('KMeans Clustering of Vehicles')
plt.show()