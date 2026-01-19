"""
NumPy Module - Array Operations and Linear Algebra
This module contains examples and utilities for working with NumPy arrays
"""

import numpy as np


def array_basics():
    """Demonstrate basic NumPy array creation and operations"""
    # Creating arrays
    a = np.array([1, 2, 3])
    print("1D Array:", a)
    
    # Multi-dimensional array
    b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
    print("2D Array:\n", b)
    
    # Get dimension
    print("Dimension of a:", a.ndim)
    print("Dimension of b:", b.ndim)
    
    # Get shape
    print("Shape of a:", a.shape)
    print("Shape of b:", b.shape)
    
    # Get type
    print("Type of a:", a.dtype)
    print("Type of b:", b.dtype)
    
    # Get size
    print("Item size of a:", a.itemsize)
    print("Item size of b:", b.itemsize)
    
    # Get total size
    print("Total size of a:", a.nbytes)
    print("Total size of b:", b.nbytes)


def array_indexing_slicing():
    """Demonstrate array indexing and slicing"""
    a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    print("Original array:\n", a)
    
    # Get specific element [r, c]
    print("Element at [0, 5]:", a[0, 5])
    print("Element at [1, -2]:", a[1, -2])
    
    # Get specific row
    print("Row 0:", a[0, :])
    
    # Get specific column
    print("Column 2:", a[:, 2])
    
    # Slicing [startindex:endindex:stepsize]
    print("Slice [0, 1:6:2]:", a[0, 1:6:2])
    
    # Modify elements
    a[1, 5] = 20
    print("Modified array:\n", a)
    
    # Modify column
    a[:, 2] = [1, 2]
    print("Modified column:\n", a)


def array_initialization():
    """Demonstrate different ways to initialize arrays"""
    # All 0s matrix
    zeros = np.zeros((2, 3))
    print("Zeros matrix:\n", zeros)
    
    # All 1s matrix
    ones = np.ones((4, 2, 2), dtype='int32')
    print("Ones matrix:\n", ones)
    
    # Any other number
    full = np.full((2, 2), 99)
    print("Full matrix:\n", full)
    
    # Full like (same shape as another array)
    a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    full_like = np.full_like(a, 4)
    print("Full like:\n", full_like)
    
    # Random decimal numbers
    rand = np.random.rand(4, 2)
    print("Random decimals:\n", rand)
    
    # Random integer values
    randint = np.random.randint(-4, 8, size=(3, 3))
    print("Random integers:\n", randint)
    
    # Identity matrix
    identity = np.identity(5)
    print("Identity matrix:\n", identity)
    
    # Repeat array
    arr = np.array([[1, 2, 3]])
    repeated = np.repeat(arr, 3, axis=0)
    print("Repeated array:\n", repeated)


def mathematics_operations():
    """Demonstrate mathematical operations on arrays"""
    a = np.array([1, 2, 3, 4])
    print("Original array:", a)
    
    # Basic operations
    print("a + 2:", a + 2)
    print("a - 2:", a - 2)
    print("a * 2:", a * 2)
    print("a / 2:", a / 2)
    print("a ** 2:", a ** 2)
    
    # Element-wise operations
    b = np.array([1, 0, 1, 0])
    print("a + b:", a + b)
    print("a * b:", a * b)


def linear_algebra():
    """Demonstrate linear algebra operations"""
    a = np.ones((2, 3))
    print("Matrix a:\n", a)
    
    b = np.full((3, 2), 2)
    print("Matrix b:\n", b)
    
    # Matrix multiplication
    c = np.matmul(a, b)
    print("Matrix multiplication (a @ b):\n", c)
    
    # Determinant
    c = np.identity(3)
    det = np.linalg.det(c)
    print("Determinant of identity matrix:", det)


def statistics_operations():
    """Demonstrate statistical operations"""
    stats = np.array([[1, 2, 3], [4, 5, 6]])
    print("Array:\n", stats)
    
    print("Min:", np.min(stats))
    print("Max:", np.max(stats))
    print("Min along axis 0:", np.min(stats, axis=0))
    print("Max along axis 1:", np.max(stats, axis=1))
    print("Sum:", np.sum(stats))
    print("Sum along axis 0:", np.sum(stats, axis=0))


def reorganizing_arrays():
    """Demonstrate array reshaping and stacking"""
    before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print("Before reshape:\n", before)
    
    after = before.reshape((4, 2))
    print("After reshape:\n", after)
    
    # Vertical stacking
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([5, 6, 7, 8])
    vstacked = np.vstack([v1, v2, v1, v2])
    print("Vertical stack:\n", vstacked)
    
    # Horizontal stacking
    h1 = np.ones((2, 4))
    h2 = np.zeros((2, 2))
    hstacked = np.hstack((h1, h2))
    print("Horizontal stack:\n", hstacked)


if __name__ == "__main__":
    print("=" * 50)
    print("NUMPY MODULE DEMONSTRATIONS")
    print("=" * 50)
    
    print("\n1. Array Basics:")
    print("-" * 50)
    array_basics()
    
    print("\n2. Array Indexing and Slicing:")
    print("-" * 50)
    array_indexing_slicing()
    
    print("\n3. Array Initialization:")
    print("-" * 50)
    array_initialization()
    
    print("\n4. Mathematics Operations:")
    print("-" * 50)
    mathematics_operations()
    
    print("\n5. Linear Algebra:")
    print("-" * 50)
    linear_algebra()
    
    print("\n6. Statistics Operations:")
    print("-" * 50)
    statistics_operations()
    
    print("\n7. Reorganizing Arrays:")
    print("-" * 50)
    reorganizing_arrays()
