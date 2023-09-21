import numpy as np
import pytest
from sparse_recommender import SparseMatrix


# Test initialization of SparseMatrix
def test_sparse_matrix_init():
    sparse_matrix = SparseMatrix([[0, 1, 0], [0, 1, 0], [0, 0, 8], [0, 0, 0]], 6, 3)
    assert isinstance(sparse_matrix, SparseMatrix)


# Test setting and getting values in the SparseMatrix
def test_set_get():
    sparse_matrix = SparseMatrix([[0, 1, 0], [0, 1, 0], [0, 0, 8], [0, 0, 0]], 6, 3)
    sparse_matrix.set(1, 2, 5)
    assert sparse_matrix.get(1, 2) == 5

# Test converting SparseMatrix to dense matrix
def test_to_dense():
    sparse_matrix = SparseMatrix([[0, 1, 0], [0, 1, 0], [0, 0, 8], [0, 0, 0]], 6, 3)
    dense_matrix = sparse_matrix.to_dense()
    expected_dense_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 9], [0, 6, 0]])
    assert np.array_equal(dense_matrix, expected_dense_matrix)

# Test for invalid key when getting a value
def test_get_invalid_key():
    sparse_matrix = SparseMatrix([[0, 0, 0], [0, 1, 0], [0, 0, 9], [0, 6, 0]], 4, 3)
    result = sparse_matrix.get(5, 5)
    assert result == "Invalid Key"

def test_set_get_value():
    sparse_matrix = SparseMatrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 3, 3)
    sparse_matrix.set(2, 1, 5)
    assert sparse_matrix.get(2, 1) == 5

# Test for invalid shape when multiplying with a vector
def test_recommend_invalid_shape():
    sparse_matrix = SparseMatrix([[0, 0, 0], [0, 1, 0], [0, 0, 9], [0, 6, 0]], 4, 3)
    invalid_vector = [1, 4, 3, 2]
    result = sparse_matrix.recommend(invalid_vector)
    assert result == "Recommendation can't be done: Invalid shape"

# Test for invalid shape when adding a matrix
def test_add_movie_invalid_shape():
    sparse_matrix = SparseMatrix([[0, 0, 0], [0, 1, 0], [0, 0, 9], [0, 6, 0]], 4, 3)
    invalid_matrix = [[0, 1, 0], [0, 0, 3], [5, 0, 9]]
    result = sparse_matrix.add_movie(invalid_matrix)
    assert result == "Move Addition Can't be Done: Invalid Shape"

def test_to_dense_matrix():
    sparse_matrix = SparseMatrix([[0, 0, 2], [0, 1, 0], [0, 0, 0]], 3, 3)
    dense_matrix = sparse_matrix.to_dense()
    expected_dense_matrix = [[0, 0, 2], [0, 1, 0], [0, 0, 0]]
    assert np.array_equal(dense_matrix, expected_dense_matrix)


