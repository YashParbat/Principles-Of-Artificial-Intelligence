import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# Adjust path accordingly
PATH = r"C:/Users/yashj/OneDrive/Documents/Yash Documents/AI Principles/UCI HAR Dataset/"

features = pd.read_csv(PATH + "features.txt", sep=r"\s+", header=None, names=["idx", "feature"])

#Deduplicate Names
def deduplicate(names):
    seen = {}
    unique_names = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique_names.append(name)
        else:
            unique_names.append(f"{name}_{seen[name]}")
            seen[name] += 1
    return unique_names

# Apply the deduplication function
feature_names = deduplicate(features["feature"])


# Load datasets
X_train = pd.read_csv(PATH + "train/X_train.txt", sep=r'\s+', header=None, names=feature_names)
y_train = pd.read_csv(PATH + "train/y_train.txt", header=None, names=["Activity"])
X_test = pd.read_csv(PATH + "test/X_test.txt", sep=r'\s+', header=None, names=feature_names)
y_test = pd.read_csv(PATH + "test/y_test.txt", header=None, names=["Activity"])

activity_labels = pd.read_csv(PATH + "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels.id, activity_labels.activity))
y_train['Activity'] = y_train['Activity'].map(activity_map)
y_test['Activity'] = y_test['Activity'].map(activity_map)

# Binary Conversion
def binary_label(activity):
    return 1 if activity in ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'] else 0

y_train_binary = y_train['Activity'].apply(binary_label)
y_test_binary = y_test['Activity'].apply(binary_label)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('svc', SVC())
])

param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10]},
    {'svc__kernel': ['poly'], 'svc__C': [0.1, 1], 'svc__degree': [2, 3], 'svc__gamma': [0.01, 0.1]},
    {'svc__kernel': ['rbf'], 'svc__C': [1, 10], 'svc__gamma': [0.001, 0.01]}
]

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train_binary)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

#Visualizing the HAR dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Reduce to 2 principal components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(X_pca[y_train_binary==0, 0], X_pca[y_train_binary==0, 1], alpha=0.5, label='Inactive', c='red')
plt.scatter(X_pca[y_train_binary==1, 0], X_pca[y_train_binary==1, 1], alpha=0.5, label='Active', c='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of HAR Data (Active vs Inactive)')
plt.legend()
plt.grid()
plt.show()
