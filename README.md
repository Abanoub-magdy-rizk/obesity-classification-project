# 🧠 Obesity Classification Using Machine Learning and Genetic Algorithm (GA)

This project aims to predict obesity levels based on health, nutrition, and lifestyle attributes using machine learning algorithms. A Genetic Algorithm (GA) was applied for dimensionality reduction to improve the predictive performance of the models.

---

## 📌 Dataset Information

- **Source:** Kaggle — Obesity Level Prediction Dataset  
- **Samples:** 2,111  
- **Original Features:** 16 (expanded to 33 after preprocessing and encoding)  
- **Task:** Multi-class classification (7 obesity classes)

### Features Include:
Age, Gender, Height, Weight, CH2O (daily water intake), SMOKE, FAF (physical activity), FCVC (vegetable intake), NCP (meals per day), MTRANS (transportation), and more.

---

## 🛠️ Data Preprocessing

- Missing numerical values → median imputation  
- Missing categorical values → mode imputation  
- One-Hot Encoding for all categorical variables  
- Feature engineering → **Body Mass Index (BMI)**  
- Standardization for numerical features  
- Target encoding and **80/20 stratified train-test split**

After preprocessing, the number of attributes increased from **16 to 33 features**.

---

## 🤖 Machine Learning Models

### ✔ Artificial Neural Network (ANN)

- hidden_layer_sizes = (48, 24)  
- activation = "relu"  
- solver = "adam"  
- alpha = 0.0001  
- early_stopping = True  
- max_iter = 800  

### ✔ Support Vector Machine (SVM)

- kernel = "rbf"  
- C = 1.0  
- gamma = "scale"  

---

## 📊 Performance on Original Dataset (Before GA)

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|--------:|
| ANN | 0.9219 | 0.9238 | 0.9220 | 0.9223 |
| SVM | 0.9480 | 0.9488 | 0.9480 | 0.9483 |

---

## 🧬 Genetic Algorithm (Feature Selection)

**GA Hyperparameters:**

- Population Size = 40  
- Generations = 20  
- Crossover Probability = 0.7  
- Mutation Probability = 0.1  
- Fitness = model validation accuracy  

**Chromosome Type:** binary vector (1 = keep feature, 0 = drop feature)

---

## ✂️ Features Selected by GA

### ✔ ANN — Selected 18 Features
Height, Weight, NCP, CH2O, BMI, Gender_Female, Gender_Male, CALC_Always,  
FAVC_no, FAVC_yes, SCC_no, SCC_yes, SMOKE_no, SMOKE_yes,  
CAEC_Always, CAEC_Frequently, MTRANS_Motorbike, MTRANS_Walking  


### ✔ SVM — Selected 22 Features
Age, Height, Weight, FCVC, NCP, CH2O, TUE, BMI, Gender_Female, Gender_Male,  
CALC_Always, CALC_Sometimes, FAVC_no, SCC_no, family_history_with_overweight_no,  
CAEC_Always, CAEC_Sometimes, CAEC_no, MTRANS_Bike, MTRANS_Motorbike,  
MTRANS_Public_Transportation, MTRANS_Walking  


---

## 📈 Performance on GA-Reduced Dataset (After GA)

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|--------:|
| ANN (After GA) | 0.9504 | 0.9498 | 0.9504 | 0.9499 |
| SVM (After GA) | 0.9622 | 0.9625 | 0.9622 | 0.9622 |

> 📌 Performance values obtained from experimental results.  
> ANN and SVM both improved after feature selection using GA. 

---

## ✔️ Key Insights

- GA reduced dimensionality by **45% for ANN** and **33% for SVM**.  
- ANN accuracy improved by **+2.84%**, SVM by **+1.42%**.  
- Eliminating redundant features improved model stability and reduced misclassification.
- **Feature quality matters more than feature quantity.**

---

## 📁 Project Files

