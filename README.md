# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset using pandas and display the first few rows to understand the data.

2. Select relevant features (Height and Weight) from the dataset for analysis.

3. Standardize the data using StandardScaler to normalize feature values.

4. Apply Principal Component Analysis (PCA) to reduce the dataset into principal components.

5. Analyze explained variance and visualize the transformed data using a scatter plot.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: R VENKATRAMANI
RegisterNumber:  212225240182 // 25010118
*/


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('HeightsWeights.csv')

print("First 5 rows of the dataset:")
print(data.head())

X = data[['Height(Inches)', 'Weight(Pounds)']]

plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title('Original Data Distribution')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="927" height="606" alt="image" src="https://github.com/user-attachments/assets/7d95e4e4-2d0f-4351-a839-856a32bc58ca" />
<img width="771" height="514" alt="image" src="https://github.com/user-attachments/assets/00057c55-da31-48cb-9db6-34ec4dbef214" />



## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
