import numpy as np
from scipy.sparse import csr_matrix


class SparseMatrix:

    def __init__(self, sparse_matrix, row, col):
        self.sparse_matrix = sparse_matrix
        self.row = row
        self.col = col

    # Empty Dictionary
    dict_sp = {}

    # Method 1 - To set Non zero elements into dictionary
    def set(self, row, col, value):
        if (row, col) not in self.dict_sp and value != 0:
            self.dict_sp[(row, col)] = value

    # Method 2 - To get the value from dictionary
    def get(self, row, col):
        if (row, col) in self.dict_sp:
            return self.dict_sp[(row, col)]
        else:
            return 'Invalid Key'

    # Method 3 - To multiply our Sparse Matrix with a user provided vector
    def recommend(self, vector):
        # Checking if both are of same shape
        if len(self.sparse_matrix[0]) == len(vector):
            res_mul = np.zeros(len(self.sparse_matrix))
            for (row, col), value in self.dict_sp.items():
                res_mul[row] = res_mul[row] + value * vector[col]
            return np.round(res_mul).astype(int).tolist()
        else:
            return "Recommendation can't be done: Invalid shape"

    # Method 4 - To Add our Sparse Matrix to other Sparse Matrix
    def add_movie(self, matrix):
        if len(matrix) == len(self.sparse_matrix) and len(matrix[0]) == len(self.sparse_matrix[0]):
            self.res_add = np.zeros((self.row, self.col))
            for i in range(self.row):
                for j in range(self.col):
                    if matrix[i][j] == 0:
                        if (i, j) in self.dict_sp:
                            self.res_add[i][j] = self.sparse_matrix[i][j]
                        else:
                            self.res_add[i][j] = 0
                    else:
                        if (i, j) in self.dict_sp:
                            self.res_add[i][j] = self.sparse_matrix[i][j] + matrix[i][j]
                        else:
                            self.res_add[i][j] = matrix[i][j]

            return self.res_add
        else:
            return "Move Addition Can't be Done: Invalid Shape"

    # Method 5 - To convert sparse matrix to dense
    def to_dense(self):
        dense_matrix = csr_matrix(self.sparse_matrix).toarray()
        return dense_matrix


'''
 Let's start with creating a Sparse Matrix to work with all our Methods
'''

sp1 = [[0, 0, 0],
       [0, 1, 0],
       [0, 0, 9],
       [0, 6, 0]]
sp2 = [[0, 1, 0],
       [0, 0, 3],
       [5, 0, 9],
       [0, 0, 0]]

vt = [1, 4, 3]
row = len(sp1)
col = len(sp1[0])

obj = SparseMatrix(sp1, row, col)

# Let's set the values
for i in range(row):
    for j in range(col):
        obj.set(i, j, sp1[i][j])
print("All Non Zeros are in Dictionary now!")

# Let's get the values from dict
print("The Value You are asking for is: ", obj.get(2, 2))

# Let's multiply the sparse with vector
print("Recommendation done:\n", obj.recommend(vt))

# Let's Add a matrix
print("Addition of a new movie done:\n", obj.add_movie(sp2))

# Let's convert sparse to dense
print("Sparse to dense done:\n", obj.to_dense())
