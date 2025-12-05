import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"..\..\..\data\processed_dataset (1).csv")

X = df.drop("target", axis=1)
y = df["target"]

print("Features used before GA:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

ann = MLPClassifier(
    hidden_layer_sizes=(48, 24),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    early_stopping=True,
    max_iter=800,
    random_state=30
)

ann.fit(X_train, y_train)
y_pred = ann.predict(X_test)

print("\n==== MODEL METRICS ====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1:", f1_score(y_test, y_pred, average='weighted'))

cm = confusion_matrix(y_test, y_pred)
cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

labels = np.array(["{}\n{:.1f}%".format(a, b)
                   for a, b in zip(cm.flatten(), cm_p.flatten())]).reshape(cm.shape)

plt.figure(figsize=(8,6))
sns.heatmap(cm_p, annot=labels, fmt="", cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ANN Before GA")
plt.show()