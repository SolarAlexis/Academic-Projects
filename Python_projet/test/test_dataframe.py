import numpy as np

from linearmodel.dataframe import Matrix

mat1 = Matrix([[1,1,2], [1,2,1], [2,1,1]])
mat2 = Matrix([[1,7,3], [1,4,5], [9,5,7]])
mat3 = np.array(mat1.values)
mat4 = np.array(mat2.values)

def test_init():
    assert (mat1.values == mat3).all()
    assert (mat2.values == mat4).all()
    assert mat1.shape == mat3.shape
    assert mat2.shape == mat4.shape
    
def test_mult():
    assert ((mat1 * mat2).values == mat3 * mat4).all()
    
def test_add():
    assert ((mat1 + mat2).values == mat3 + mat4).all()
    
def test_dot():
    assert ((mat1.dot(mat2)).values == mat3 @ mat4).all()
    
def test_transpose():
    assert ((mat1.transpose()).values == mat3.T).all()
    assert ((mat2.transpose()).values == mat4.T).all()
    
def test_getitem():
    assert (mat1[0] == mat3[0]).all()
    assert (mat1[:,1] == mat3[:,1]).all()
    assert (mat1[1][2] == mat3[1][2]).all()
    assert (mat1[2,:1] == mat3[2,:1]).all()
    
def test_determinant():
    assert mat1.determinant() == np.linalg.det(mat3)

def test_inv():
    assert str(np.array(mat1.inv().values)) == str(np.linalg.inv(mat3))
    
if __name__ == "__main__":
    test_init()
    test_mult()
    test_add()
    test_dot()
    test_transpose()
    test_getitem()
    test_determinant()
    test_inv()