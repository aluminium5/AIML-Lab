# 🤖 AI & ML Lab Experiments

Welcome to the **AI-ML-LAB** repository! This project contains a collection of fundamental Artificial Intelligence and Machine Learning experiments, implemented in Python. It serves as a practical guide to understanding core concepts, algorithms, and data manipulation techniques.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📂 Repository Structure

The experiments are organized by topic within the `ai_ml_lab_experiments` directory:

| File | Topic | Description |
| :--- | :--- | :--- |
| **`01_numpy_intro.py`** | **Numpy** | Introduction to array creation, broadcasting, and performance benefits over lists. |
| **`02_pandas_intro.py`** | **Pandas** | Basics of DataFrames, reading CSVs/Excel, and data cleaning techniques. |
| **`03_ecommerce_analysis.py`** | **EDA** | Exploratory Data Analysis (EDA) on an Ecommerce dataset to answer business questions. |
| **`04_logistic_regression_ads.py`** | **Regression** | Predicting user purchases from "Social Network Ads" using Logistic Regression. |
| **`05_classification_pipeline...`** | **Pipeline** | End-to-end pipeline (Preprocessing -> PCA -> Naive Bayes/KNN) on the "Life Style" dataset. |
| **`06_knn_iris.py`** | **Classification** | K-Nearest Neighbors (KNN) implementation on the classic Iris dataset. |
| **`07_naive_bayes_iris.py`** | **Classification** | Gaussian Naive Bayes implementation on the Iris dataset. |
| **`08_kmeans_iris.py`** | **Clustering** | Unsupervised learning using K-Means clustering to group Iris flowers. |
| **`09_pca_iris.py`** | **Dim. Reduction** | Visualizing high-dimensional data using Principal Component Analysis (PCA). |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running Experiments

Navigate to the experiment directory and run a script:

```bash
cd ai_ml_lab_experiments
python 06_knn_iris.py
```

---

## 📊 Datasets

To run these scripts successfully, you will need to download the following datasets from Kaggle and place them in the `ai_ml_lab_experiments` folder.

| Dataset Name | Source | Required File Name | Used In |
| :--- | :--- | :--- | :--- |
| **Pokemon Dataset** | [Kaggle Link](https://www.kaggle.com/datasets/abcsds/pokemon) | `pokemon_data.csv` | `02_pandas_intro.py` |
| **Ecommerce Purchases** | [Kaggle Link](https://www.kaggle.com/datasets/jmmvutu/ecommerce-purchases) | `Ecommerce Purchases.csv` | `03_ecommerce_analysis.py` |
| **Social Network Ads** | [Kaggle Link](https://www.kaggle.com/datasets/rakeshrau/social-network-ads) | `Social_Network_Ads1.csv` | `04_logistic_regression_ads.py` |
| **Life Style Dataset** | [Kaggle Link](https://www.kaggle.com/datasets/aditya08/life-style-dataset) | `Final_data.csv` | `05_classification_pipeline...` |

> **Note:** Please rename the downloaded CSV files to match the "Required File Name" column if they differ.

---

## 🤝 Contributing

Feel free to fork this repository and submit Pull Requests if you'd like to add more experiments or improve existing ones!
