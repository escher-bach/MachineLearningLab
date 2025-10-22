import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import KNNClassifier
from datawine import X, y
from utils import train_test_split

# Split the data into training and testing sets
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten y arrays for easier comparison
y_train = y_train.flatten()
y_test = y_test.flatten()

# 1. Train and evaluate with k=3
print("=" * 50)
print("KNN Classification with k=3")
print("=" * 50)

knn = KNNClassifier(k=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
print()

# 2. Hyperparameter tuning
print("=" * 50)
print("Hyperparameter Tuning")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    accuracies.append(accuracy)
    
    print(f"k={k:2d} | Accuracy: {accuracy * 100:.2f}%")

# 3. Plot accuracy vs k-value
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('k-value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN Classification: Accuracy vs k-value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.ylim([0, 1.1])

# Add value labels on points
for k, acc in zip(k_values, accuracies):
    plt.text(k, acc + 0.02, f'{acc * 100:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print()
print("=" * 50)
print(f"Best k-value: {k_values[np.argmax(accuracies)]} with accuracy: {max(accuracies) * 100:.2f}%")
print("=" * 50)
