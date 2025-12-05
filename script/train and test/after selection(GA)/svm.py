import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from deap import base, creator, tools, algorithms
import random

df = pd.read_csv(r"..\..\..\data\processed_dataset (1).csv")

X = df.drop("target", axis=1)
y = df["target"]

print("Features BEFORE GA:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

POP_SIZE = 40
N_GEN = 20
MUT_PB = 0.1
CX_PB = 0.7


def evaluate(individual):
    mask = np.array(individual, dtype=bool)

    if sum(mask) == 0:
        return 0.0,

    X_tr_sub = X_tr.iloc[:, mask]
    X_val_sub = X_val.iloc[:, mask]

    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_tr_sub, y_tr)
    y_pred = model.predict(X_val_sub)
    acc = accuracy_score(y_val, y_pred)

    return acc,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUT_PB)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=POP_SIZE)

result, log = algorithms.eaSimple(
    population, toolbox,
    cxpb=CX_PB,
    mutpb=MUT_PB,
    ngen=N_GEN,
    verbose=True
)

best_ind = tools.selBest(population, k=1)[0]
mask = np.array(best_ind, dtype=bool)

print("\nBest Chromosome:", best_ind)
print("Features AFTER GA:", sum(mask))
print("Reduced from {} to {}".format(X_train.shape[1], sum(mask)))

selected_features = X_train.columns[mask]
print("\n=== SELECTED FEATURES (AFTER GA) ===")
for f in selected_features:
    print(f)

print("\nTotal Selected Features:", len(selected_features))

X_train_sel = X_train.iloc[:, mask]
X_test_sel = X_test.iloc[:, mask]

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_sel, y_train)
y_pred = model.predict(X_test_sel)

print("\n==== MODEL METRICS (AFTER GA — TEST SET) ====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

cm = confusion_matrix(y_test, y_pred)
cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

labels = np.array([
    "{}\n{:.1f}%".format(a, b)
    for a, b in zip(cm.flatten(), cm_p.flatten())
]).reshape(cm.shape)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_p, annot=labels, fmt="", cmap="Blues",
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test)
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (AFTER GA, No Data Leakage)")
plt.show()