import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np 

customer_data = pd.read_csv("customers.csv")
X = customer_data[["annual_income_k", "spending_score"]]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
customer_data["Cluster"] = kmeans.labels_
print(customer_data)
print("Cluster Centers:")
print(kmeans.cluster_centers_)

plt.scatter(
    customer_data["annual_income_k"],
    customer_data["spending_score"],
    c=customer_data["Cluster"],
    cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X",
    color="red"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()

while True:
    a = int(input("Enter Annual Income (k$): "))
    b = int(input("Enter Spending Score (1-100): "))

    new_customer = pd.DataFrame(
        [[a, b]],
        columns=["annual_income_k", "spending_score"]
    )

    cluster = kmeans.predict(new_customer)

    if cluster[0] == 2:
        print("UDHARI")
    elif cluster[0] == 1:
        print("SELEVALI")
    else:
        print("KANJA PISNARI")

    print("Cluster:", cluster[0])