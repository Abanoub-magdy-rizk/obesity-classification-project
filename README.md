# 🧠 Obesity Level Prediction Using Machine Learning & Genetic Algorithm

A complete end-to-end machine learning workflow designed to classify obesity levels based on lifestyle, physical characteristics, and dietary habits. The project also applies a **Genetic Algorithm (GA)** for feature selection to reduce dimensionality and improve model performance.

---

## 📌 Dataset Overview

- **Source:** Kaggle – Obesity Level Prediction Dataset  
- **Samples:** 2,111  
- **Original Features:** 16  
- **Target Classes:** 7  
- **Final Features After Preprocessing:** 33  

### Target Classes:
- Insufficient Weight  
- Normal Weight  
- Overweight Level I  
- Overweight Level II  
- Obesity Type I  
- Obesity Type II  
- Obesity Type III  

---

## 🛠️ Data Preprocessing

- Missing numerical values → **Median imputation**  
- Missing categorical values → **Most frequent value**  
- **One-Hot Encoding** for categorical attributes  
- **Feature engineering:** BMI (Body Mass Index) calculated and added  
- Numerical features standardized to zero mean and unit variance  
- Target variable label-encoded  
- **Train/Test split:** 80% training, 20% testing (stratified)

---

## 🤖 Machine Learning Models

### ✔️ Artificial Neural Network (ANN)

- hidden_layer_sizes = (48, 24)  
- activation = "relu"  
- solver = "adam"  
- alpha = 0.0001  
- early_stopping = True  
- max_iter = 800  

### ✔️ Support Vector Machine (SVM)

- kernel = "rbf"  
- C = 1.0  
- gamma = "scale"  

---

## 📊 Performance (Original Dataset — Before GA)

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|--------:|
| ANN  | 0.9219 | 0.9238 | 0.9220 | 0.9223 |
| SVM  | 0.9480 | 0.9488 | 0.9480 | 0.9483 |

---

## 🧬 GA-Based Feature Selection

### Genetic Algorithm Parameters

- Population Size (POP_SIZE): 40  
- Number of Generations (N_GEN): 20  
- Crossover Probability (CX_PB): 0.7  
- Mutation Probability (MUT_PB): 0.1  
- Selection Method: Tournament (size = 3)  
- Fitness Function: Validation accuracy  

### Chromosome Representation
Binary vector (1 = keep feature, 0 = remove feature)  
Length = 33 features  

---

## ✂️ Feature Reduction Results

| Model | Features Before | Features After | Reduction |
|------|----------------:|---------------:|---------:|
| ANN  | 33 | 18 | ~45% |
| SVM  | 33 | 22 | ~33% |

---

## 📈 Performance (After GA — Reduced Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|--------:|
| ANN (After GA) | 0.9504 | 0.9498 | 0.9504 | 0.9499 |
| SVM (After GA) | 0.9622 | 0.9625 | 0.9622 | 0.9622 |

---

## ✔️ Key Findings

- Performance **improved for both ANN and SVM** after applying GA.  
- Reducing the number of features improved generalization and classification stability.  
- SVM achieved the **best results overall**, both before and after GA.  
- Dimensionality reduction removed weak/irrelevant features and focused on the most informative ones.

> 🧩 Important Insight:  
> **More features ≠ better performance**.  
> The models performed better when fewer, more meaningful features were used.

---

## 📁 Project Structure

