# segmentation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# ----- Paths -----
DATA_PATH = Path(__file__).parents[1] / "data" / "Mall_Customers.csv"
OUT_DIR = Path(__file__).parents[1] / "output"
OUT_DIR.mkdir(exist_ok=True)

# ----- Load and Preprocess -----
df = pd.read_csv(DATA_PATH)
df = df.drop("CustomerID", axis=1)

le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])  # Male = 1, Female = 0

scaler = StandardScaler()
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns)

# ----- EDA Plots -----
sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

sns.countplot(x="Genre", data=df, ax=axs[0, 0])
axs[0, 0].set_title("Gender Count")

sns.histplot(df["Age"], bins=15, kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Age Distribution")

sns.histplot(df["Annual Income (k$)"], bins=15, kde=True, ax=axs[0, 2], color="green")
axs[0, 2].set_title("Income Distribution")

sns.histplot(df["Spending Score (1-100)"], bins=15, kde=True, ax=axs[1, 0], color="orange")
axs[1, 0].set_title("Spending Score")

sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Genre", data=df, ax=axs[1, 1])
axs[1, 1].set_title("Income vs Spending")

sns.scatterplot(x="Age", y="Spending Score (1-100)", hue="Genre", data=df, ax=axs[1, 2])
axs[1, 2].set_title("Age vs Spending")

plt.tight_layout()
plt.savefig(OUT_DIR / "eda_plots.png")
plt.close()

# ----- Elbow Method -----
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=1)
    km.fit(scaled_df)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig(OUT_DIR / "elbow.png")
plt.close()

# ----- KMeans Clustering (Set k based on elbow.png) -----
k = 5
km_final = KMeans(n_clusters=k, random_state=1)
df["Cluster"] = km_final.fit_predict(scaled_df)

# ----- Cluster Plot -----
plt.figure(figsize=(9, 6))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=df, palette="Set2", s=100)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.savefig(OUT_DIR / "clusters.png")
plt.close()

# ----- Save Output -----
df.to_csv(OUT_DIR / "clustered_customers.csv", index=False)

# Print basic summary
print("\n=== Cluster Averages ===\n")
print(df.groupby("Cluster").mean(numeric_only=True))
