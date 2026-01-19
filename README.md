# 🤖 AI & ML Lab Experiments

Welcome to the **AIML-Lab** repository! This project contains a comprehensive collection of Artificial Intelligence and Machine Learning experiments, implemented in Python. It serves as a practical guide to understanding core concepts, algorithms, and data manipulation techniques.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## � Repository Structure

The experiments are organized by topic with both individual scripts and organized modules:

### **Core Lab Experiments**

| File | Topic | Description |
| :--- | :--- | :--- |
| **`01_numpy_intro.py`** | **NumPy** | Introduction to array creation, broadcasting, and performance benefits over lists. |
| **`02_pandas_intro.py`** | **Pandas** | Basics of DataFrames, reading CSVs/Excel, and data cleaning techniques. |
| **`03_ecommerce_analysis.py`** | **EDA** | Exploratory Data Analysis (EDA) on an Ecommerce dataset to answer business questions. |
| **`04_logistic_regression_ads.py`** | **Regression** | Predicting user purchases from "Social Network Ads" using Logistic Regression. |
| **`05_classification_pipeline_final_data.py`** | **Pipeline** | End-to-end pipeline (Preprocessing → PCA → Naive Bayes/KNN) on the "Life Style" dataset. |
| **`06_knn_iris.py`** | **Classification** | K-Nearest Neighbors (KNN) implementation on the classic Iris dataset. |
| **`07_naive_bayes_iris.py`** | **Classification** | Gaussian Naive Bayes implementation on the Iris dataset. |
| **`08_kmeans_iris.py`** | **Clustering** | Unsupervised learning using K-Means clustering to group Iris flowers. |
| **`09_pca_iris.py`** | **Dim. Reduction** | Visualizing high-dimensional data using Principal Component Analysis (PCA). |

### **Organized Modules**

| File | Description |
| :--- | :--- |
| **`numpy_module.py`** | Comprehensive NumPy demonstrations with organized functions covering array operations, linear algebra, and statistics. |
| **`pandas_module.py`** | Complete Pandas demonstrations including data reading, exploration, filtering, sorting, modification, and aggregation. |
| **`main.py`** | Interactive menu system to run demonstrations from both NumPy and Pandas modules. |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running Experiments

**Option 1: Interactive Menu (Recommended)**
```bash
python main.py
```
This will display an interactive menu where you can choose which module to run.

**Option 2: Run Individual Lab Experiments**
```bash
# Run any specific experiment
python 01_numpy_intro.py
python 06_knn_iris.py
```

**Option 3: Run Organized Modules**
```bash
# Run NumPy demonstrations
python numpy_module.py

# Run Pandas demonstrations
python pandas_module.py
```

---

## � Datasets

To run these scripts successfully, you will need to download the following datasets from Kaggle and place them in the project folder.

| Dataset Name | Source | Required File Name | Used In |
| :--- | :--- | :--- | :--- |
| **Pokemon Dataset** | [Kaggle Link](https://www.kaggle.com/datasets/abcsds/pokemon) | `pokemon_data.csv` | `02_pandas_intro.py`, `pandas_module.py` |
| **Ecommerce Purchases** | [Kaggle Link](https://www.kaggle.com/datasets/jmmvutu/ecommerce-purchases) | `Ecommerce Purchases.csv` | `03_ecommerce_analysis.py` |
| **Social Network Ads** | [Kaggle Link](https://www.kaggle.com/datasets/rakeshrau/social-network-ads) | `Social_Network_Ads1.csv` | `04_logistic_regression_ads.py` |
| **Life Style Dataset** | [Kaggle Link](https://www.kaggle.com/datasets/aditya08/life-style-dataset) | `Final_data.csv` | `05_classification_pipeline_final_data.py` |

> **Note:** Please rename the downloaded CSV files to match the "Required File Name" column if they differ.

---

## 📚 Module Details

### **numpy_module.py**
Comprehensive NumPy demonstrations including:
- ✅ Array basics (creation, dimensions, shape, type)
- ✅ Array indexing and slicing
- ✅ Array initialization (zeros, ones, random, identity)
- ✅ Mathematical operations and broadcasting
- ✅ Linear algebra (matrix multiplication, determinants)
- ✅ Statistical operations (min, max, sum, mean)
- ✅ Array reorganization (reshape, stack)

### **pandas_module.py**
Complete Pandas demonstrations including:
- ✅ Reading data (CSV, Excel)
- ✅ Basic data exploration (head, tail, info, describe)
- ✅ Data filtering (conditions, string operations, regex)
- ✅ Data sorting (single and multiple columns)
- ✅ Data modification (adding columns, reordering)
- ✅ Saving data (CSV, Excel, text)
- ✅ Aggregate statistics (groupby operations)
- ✅ Working with large files (chunking)

---

## 🎯 Learning Path

**For Beginners:**
1. Start with `01_numpy_intro.py` or `numpy_module.py`
2. Move to `02_pandas_intro.py` or `pandas_module.py`
3. Try `03_ecommerce_analysis.py` for practical EDA

**For ML Enthusiasts:**
1. Explore `06_knn_iris.py` and `07_naive_bayes_iris.py` for classification
2. Learn clustering with `08_kmeans_iris.py`
3. Understand dimensionality reduction with `09_pca_iris.py`
4. Study the complete pipeline in `05_classification_pipeline_final_data.py`

---

## 🔧 Features

- **Well-Commented Code**: Every script includes detailed comments explaining the logic
- **Modular Design**: Organized functions that can be imported and reused
- **Interactive Menu**: Easy-to-use interface for running demonstrations
- **Error Handling**: Graceful handling of missing files and common errors
- **Best Practices**: Follows Python coding standards and ML best practices

---

## 🤝 Contributing

Feel free to fork this repository and submit Pull Requests if you'd like to add more experiments or improve existing ones!

---

## 📝 License

This project is open source and available for educational purposes.

---

## 👨‍💻 Author

**Atharv**
- GitHub: [@aluminium5](https://github.com/aluminium5)

---

## � Acknowledgments

- Original experiments inspired by various AI/ML courses and tutorials
- Special thanks to the open-source community for amazing libraries like NumPy, Pandas, and Scikit-learn

---

**⭐ Star this repository if you find it helpful!**
