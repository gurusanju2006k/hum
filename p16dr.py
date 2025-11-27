# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv(r"C:\Users\gurus\OneDrive\Desktop\50_Startups.csv")      # change path as needed
df = pd.DataFrame(data, columns=data['State'])
print(df.head())

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(df)
scaled_df = pd.DataFrame(X, columns=df.columns)

# Check correlation before PCA
sns.heatmap(scaled_df.corr())
plt.title("Correlation Before PCA")
plt.show()

# Applying PCA (select 3 principal components)
pca = PCA(n_components=3)
pca.fit_transform(X)
data_pca = pca.transform(X)

# Converting PCA output to DataFrame
pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
print(data_pca.head())

# Check correlation after PCA
sns.heatmap(pca_df.corr())
plt.title("Correlation After PCA")
plt.show()
