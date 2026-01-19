# AIML Lab - Python Modules

This project contains organized Python modules based on the Colab notebook for AIML Lab demonstrations.

## 📁 Project Structure

```
AIML-Lab/
├── main.py              # Main entry point with interactive menu
├── numpy_module.py      # NumPy demonstrations and utilities
├── pandas_module.py     # Pandas demonstrations and utilities
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites
Make sure you have Python installed with the following packages:
```bash
pip install numpy pandas
```

### Running the Program

**Option 1: Interactive Menu**
```bash
python main.py
```
This will display an interactive menu where you can choose which module to run.

**Option 2: Run Individual Modules**
```bash
# Run NumPy demonstrations
python numpy_module.py

# Run Pandas demonstrations
python pandas_module.py
```

## 📚 Module Descriptions

### 1. **numpy_module.py**
Contains comprehensive NumPy demonstrations including:
- ✅ Array basics (creation, dimensions, shape, type)
- ✅ Array indexing and slicing
- ✅ Array initialization (zeros, ones, random, identity)
- ✅ Mathematical operations
- ✅ Linear algebra (matrix multiplication, determinants)
- ✅ Statistical operations (min, max, sum, mean)
- ✅ Array reorganization (reshape, stack)

### 2. **pandas_module.py**
Contains comprehensive Pandas demonstrations including:
- ✅ Reading data (CSV, Excel)
- ✅ Basic data exploration (head, tail, info, describe)
- ✅ Data filtering (conditions, string operations, regex)
- ✅ Data sorting (single and multiple columns)
- ✅ Data modification (adding columns, reordering)
- ✅ Saving data (CSV, Excel, text)
- ✅ Aggregate statistics (groupby operations)
- ✅ Working with large files (chunking)

### 3. **main.py**
Interactive menu system to run demonstrations from both modules.

## 📊 Data Requirements

For the Pandas module to work properly, you need the `pokemon_data.csv` file in the same directory. If you don't have it, you can:
1. Download it from the Colab notebook
2. Use any other CSV file and modify the code accordingly
3. The module will show an error message if the file is not found

## 🎯 Usage Examples

### NumPy Module
```python
import numpy_module

# Run all demonstrations
# (automatically runs when executed as main)

# Or import specific functions
from numpy_module import array_basics, linear_algebra
array_basics()
linear_algebra()
```

### Pandas Module
```python
import pandas_module

# Run all demonstrations
pandas_module.main()

# Or import specific functions
from pandas_module import filtering_data, aggregate_statistics
df = pd.read_csv('pokemon_data.csv')
filtering_data(df)
aggregate_statistics(df)
```

## 🔧 Customization

Each module is organized with separate functions for different topics. You can:
- Import specific functions for your needs
- Modify the demonstrations to work with your own data
- Add new functions following the same pattern

## 📝 Notes

- All code is well-commented and organized
- Each function is self-contained and can be used independently
- The modules follow Python best practices
- Error handling is included for common issues

## 🤝 Contributing

Feel free to add more demonstrations or improve existing ones!

---

**Created from Colab Notebook**: [aiml lab](https://colab.research.google.com/drive/1Y_8ku5Db-tB5tmRUb0dMT5ECpDy0PFWh)
