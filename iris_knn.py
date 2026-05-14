import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
print("All libraries import successfully")
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"]= iris.target_names[iris.target]

print(f" Row (instances):{len(df)}")
print(f"   Columns (attributes): {len(df.columns)}")
print(f"   Column names: {list(df.columns)}")
df
df.describe()
print("Samples per class:")
print(df['species'].value_counts())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
plt.figure(figsize=(7, 5))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% for testing
    random_state=40     # For reproducibility
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
model = KNeighborsClassifier(n_neighbors=5) # Using 5 neighbors as a common starting point
model.fit(X_train, y_train)

print(" Model trained successfully!")
print(f"   Number of neighbors (k) : {model.n_neighbors}")
print(f"   Classes                 : {list(model.classes_)}")
y_pred = model.predict(X_test)

# Peek at the first 10 predictions vs actual
comparison = pd.DataFrame({'Actual': y_test.values[:], 'Predicted': y_pred[:]})
print(comparison.to_string(index=False))
accuracy   = sklearn.metrics.accuracy_score(y_test, y_pred)
precision  = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
recall     = sklearn.metrics.recall_score(y_test, y_pred, average='macro')
f1         = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
error_rate = 1 - accuracy

print(" Model Performance Metrics")
print("-" * 30)
print(f"  Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f}")
print(f"  F1 Score   : {f1:.4f}")
print(f"  Error Rate : {error_rate:.4f}  ({error_rate*100:.2f}%)")
print(" Classification Report")
print("-" * 50)
print(sklearn.metrics.classification_report(y_test, y_pred))
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Predicted species:", model.predict(sample))
