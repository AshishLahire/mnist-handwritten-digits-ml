# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from scipy.stats import mode

# -------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------
os.makedirs("output", exist_ok=True)

# TASK 1: DATA LOADING & EXPLORATION
# ===============================
print("\n--- Task 1: Data Loading & Exploration ---")

# Load CSV
df = pd.read_csv("C:\\Users\\ashis\\OneDrive\\Desktop\\assinment\\mnist_test.csv\\mnist_test.csv")  # make sure the CSV is in same folder

print("Total samples:", df.shape[0])
print("Total features per sample:", df.shape[1] - 1)

# Class distribution
class_counts = df["label"].value_counts().sort_index()
print("\nClass Distribution:\n", class_counts)

# Visualize class distribution
plt.figure(figsize=(8,4))

sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.savefig("output/class_distribution.png")
plt.show()

# Display 10 sample images
X = df.drop("label", axis=1).values
y = df["label"].values

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i].reshape(28,28), cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("output/sample_images.png")
plt.show()


# Check missing values
print("Total missing values:", df.isnull().sum().sum())

# ===============================
# TASK 2: DATA PREPROCESSING
# ===============================
print("\n--- Task 2: Data Preprocessing ---")

# Normalize pixel values
X = X / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA for dimensionality reduction
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Original dimension:", X_train.shape[1])
print("Reduced dimension after PCA:", X_train_pca.shape[1])

# TASK 3: MODEL TRAINING

print("\n--- Task 3: Model Training ---")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)

# SVM
svm = SVC(kernel="rbf", C=5, gamma="scale")
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# TASK 4: MODEL EVALUATION
# ===============================
print("\n--- Task 4: Model Evaluation ---")

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"output/confusion_{name.lower()}.png")
    plt.show()
    
    return cm

cm_knn = evaluate_model("KNN", y_test, y_pred_knn)
cm_svm = evaluate_model("SVM", y_test, y_pred_svm)
cm_dt = evaluate_model("DecisionTree", y_test, y_pred_dt)

# Visualize misclassified images (SVM)
mis_idx = np.where(y_test != y_pred_svm)[0][:10]
plt.figure(figsize=(10,4))
for i, idx in enumerate(mis_idx):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{y_pred_svm[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("output/misclassified_svm.png")
plt.show()

# ===============================
# TASK 5: BONUS 1 - Voting Ensemble
# ===============================
print("\n--- Bonus: Voting Ensemble ---")
ensemble_preds = mode(
    np.vstack([y_pred_knn, y_pred_svm, y_pred_dt]), axis=0
)[0].flatten()
print("Voting Ensemble Accuracy:", accuracy_score(y_test, ensemble_preds))

# ===============================
# BONUS 2: KNN FROM SCRATCH
# ===============================
print("\n--- Bonus: KNN From Scratch ---")

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN_Scratch:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        preds = []
        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_idx = np.argsort(distances)[:self.k]
            labels = self.y_train[k_idx]
            preds.append(np.bincount(labels).argmax())
        return np.array(preds)

# Train on small subset for speed
knn_scratch = KNN_Scratch(k=3)
knn_scratch.fit(X_train[:2000], y_train[:2000])
scratch_preds = knn_scratch.predict(X_test[:200])
print("KNN From Scratch Accuracy:", accuracy_score(y_test[:200], scratch_preds))

print("\n--- FULL ASSIGNMENT COMPLETED ---")