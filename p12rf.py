# 12. Implement Random Forest Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset
data = pd.read_csv(r"C:/Users/User_data.csv")

# Features: columns 2 & 3 (Experience, Salary)
x = data.iloc[:, [2, 3]].values

# Target: column 4 (Purchased)
y = data.iloc[:, 4].values

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------
#   RANDOM FOREST CLASSIFIER
# -----------------------------
classifier = RandomForestClassifier(
    n_estimators=10,         # number of trees
    criterion='entropy',    # same as decision tree criterion
    random_state=0
)
classifier.fit(x_train, y_train)

# Predictions
y_pred = classifier.predict(x_test)

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
#   PLOT DECISION BOUNDARY
# -----------------------------
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

# Plot training data
for i in np.unique(y_train):
    plt.scatter(
        x_train[y_train == i, 0],
        x_train[y_train == i, 1],
        c=ListedColormap(['purple', 'green'])(i),
        label=i
    )

plt.title('Random Forest Classifier (Training Set)')
plt.xlabel('Experience (Scaled)')
plt.ylabel('Salary (Scaled)')
plt.legend()
plt.show()
