# 11. Implement Decision Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load the dataset
data = pd.read_csv(r"C:\Users\gurus\OneDrive\Desktop\User_data.csv")

# Extract features and target
x = data.iloc[:, [2, 3]].values  # assuming columns 2 & 3 are Experience and Salary
y = data.iloc[:, 4].values       # target variable (e.g., Purchased)

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Decision Tree Classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Predict and print confusion matrix
y_pred = classifier.predict(x_test)
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Plot decision boundary
x1, x2 = np.meshgrid(
    np.arange(x_train[:, 0].min() - 1, x_train[:, 0].max() + 1, 0.01),
    np.arange(x_train[:, 1].min() - 1, x_train[:, 1].max() + 1, 0.01)
)
plt.contourf(
    x1, x2,
    classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha=0.75,
    cmap=ListedColormap(['purple', 'green'])
)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Plot training points
for j, i in enumerate(np.unique(y_train)):
    plt.scatter(
        x_train[y_train == j, 0],
        x_train[y_train == j, 1],
        c=ListedColormap(['purple', 'green'])(i),
        label=j
    )

plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('Experience (scaled)')
plt.ylabel('Salary (scaled)')
plt.legend()
plt.show()
