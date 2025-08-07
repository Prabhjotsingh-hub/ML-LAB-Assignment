                                    # LAB ASSIGNMENT 1

# Q1: Questions on Basic NumPy Array 
import numpy as np

# (a)	Reverse the NumPy array: arr = np.array([1, 2, 3, 6, 4, 5]) 
arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print("REVERSED ARRAY IS :")
print(reversed_arr)
print()

# (b)	Flatten the NumPy arr: array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]) using any two NumPy in-built methods 
arr1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])

# Method 1: ravel()
flat1 = arr1.ravel()
print("FLAT1 IS:")
print(flat1)
# Method 2: flatten()
print("FLAT2 IS:")
flat2 = arr1.flatten()
print(flat2)
print()

# (c)	Compare the following numpy arrays: 
# arr1 = np.array([[1, 2], [3, 4]]) arr2 = np.array([[1, 2], [3, 4]]) 
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

are_equal = np.array_equal(arr1, arr2)
print("ARE THEY EQUAL:")
print(are_equal)
print()
 
# (d)	Find the most frequent value and their indice(s) in the following arrays: 	
#i.	x = np.array([1,2,3,4,5,1,2,1,1,1]) 
x = np.array([1,2,3,4,5,1,2,1,1,1]) 
freq_x = np.bincount(x).argmax()
indices_x = np.where(x == freq_x)[0]
print(indices_x)

# ii.	y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3, ]) 
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
freq_y = np.bincount(y).argmax()
indices_y = np.where(y == freq_y)[0]
print(indices_y)


# (e)	For the array gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]'), find
gfg = np.matrix([[4, 1, 9], [12, 3, 1], [4, 5, 6]])
# i. Sum of all elements
sum_all = gfg.sum()
print("THE SUM OD ALL IS:")
print(sum_all)
print()

# ii.	Sum of all elements row-wise  
sum_row = gfg.sum(axis=1)
print("ROWISE SUM IS:")
print(sum_row)
print()

# iii.	Sum of all elements column-wise 
sum_col = gfg.sum(axis=0)
print("COLUMN WISE SUM IS:")
print(sum_col)
print()

# (f)	For the matrix: n_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]]), find 
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])

# i.	Sum of diagonal elements
diagonal = np.trace(n_array)
print("SUM OF DIAGONAL ELEMENTS IS:")
print(diagonal)
print()

# ii.	Eigen values of matrix 
eigen_vals,eigen_vecs = np.linalg.eig(n_array)
print("EIGEN VALUES:")
print(eigen_vals)
print()

# iii.	Eigen vectors of matrix 
print("EIGEN VECTORS:")
print(eigen_vecs)
print()

# iv. Inverse of matrix 
inverse = np.linalg.inv(n_array)
print("INVERSE OF MATRIX IS:")
print(inverse)
print()

# iv. Determinant of matrix 
det = np.linalg.det(n_array)
print("DETERMINANT OF MATRIX IS:")
print(det)
print()

# (g)	Multiply the following matrices and also find covariance between matrices using NumPy: 
# i. p = [[1, 2], [2, 3]] q = [[4, 5], [6, 7]] 
# i.
p = np.array([[1, 2], [2, 3]])
q = np.array([[4, 5], [6, 7]])
product1 = np.dot(p, q)
print("PRODUCT OF THE MATRICES IS:")
print(product1)
cov1 = np.cov(p.T, q.T)
print("COVARIANCE OF THESE TWO MATRICES IS:")
print(cov1)

# ii. 	p = [[1, 2], [2, 3], [4, 5]] q = [[4, 5, 1], [6, 7, 2]] 
p = np.array([[1, 2], [2, 3], [4, 5]])
q = np.array([[4, 5, 1], [6, 7, 2]])
product2 = np.dot(p, q)
cov2 = np.cov(p.T, q.T)

# (h)	For the matrices: x = np.array([[2, 3, 4], [3, 2, 9]]); y = np.array([[1, 5, 0], [5, 10, 3]]), find inner, outer and cartesian product? 

