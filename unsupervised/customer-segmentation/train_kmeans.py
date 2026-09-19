import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load customer dataset
customer_data = pd.read_csv("customers.csv")

# Select features
X = customer_data[["annual_income_k", "spending_score"]]

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels
customer_data["Cluster"] = kmeans.labels_

# Print results
print(customer_data)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Save trained model
joblib.dump(kmeans, "kmeans_model.pkl")

print("\nModel trained and saved successfully!")