from tqdm import tqdm

# Laoding the data in a list of list
def load(path, sep):
        
    with open(path, encoding="utf-8") as f:
        read_data = f.read()
            
    res = [read_data.split('\n')[i].split(sep) for i in tqdm(range(len(read_data.split('\n'))))]
    res = res[:len(res)-1]
        
    return res

# Dataframe used to manipulate the data
class Matrix():
    def __init__(self, values):
        self.values = values
        if type(self.values[0]) is not list:
            self.shape = (len(self.values), 1)
        else:
            self.shape = (len(self.values), len(self.values[0]))
    
    def __mul__(self, mat):
        if self.shape != mat.shape:
            raise ValueError("dimensions n and m are not equal")
        else:
            res = [[self.values[i][j] * mat.values[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(res)
    
    def __add__(self, mat):
        if self.shape != mat.shape:
            raise ValueError("dimensions n and m are not equal")
        else:
            n, m = self.shape
            if m == 1:
                res = [self.values[i] + mat.values[i] for i in range(n)]
            else:
                res = [[self.values[i][j] + mat.values[i][j] for j in range(m)] for i in range(n)]
            return Matrix(res)
    
    def __str__(self):
        n, m = self.shape
        res = ''
        
        if m == 1:
            for i in range(n):
                res += f'{self.values[i]} '
        else:
            for i in range(n):
                for j in range(m):
                    res += f'{self.values[i][j]} '
                res += '\n'  
        return res
    
    def __repr__(self):
        return self.__str__()
    
    def dot(self, mat):
        if self.shape[1] != mat.shape[0]:
            raise ValueError("Matrices are not multiplicable")
        else:
            n1, m1 = self.shape
            n2, m2 = mat.shape
            init = [[0 for i in range(m2)] for j in range(n1)]
            for i in range(n1):
                for j in range(m2):
                    init[i][j] = sum([self.values[i][k] * mat.values[k][j] for k in range(m1)])
            return Matrix(init)
        
    def transpose(self):
        n, m = self.shape
        
        res = [[self.values[j][i] for j in range(n)] for i in range(m)]
        
        return Matrix(res)
    
    def __getitem__(self, dim):
        if type(dim) == int:
            return self.values[dim]
        if (type(dim[0]) == int and type(dim[1]) == int) or (type(dim[0]) == int and type(dim[1]) == slice):
            return self.values[dim[0]][dim[1]]
        else:
            return [self.values[i][dim[1]] for i in range(self.shape[0])][dim[0]]
        
    def copy(self):
        return Matrix(self.values.copy())
    
    def __setitem__(self, row, col, value):
        if 0 <= row < len(self.values) and 0 <= col < len(self.values[0]):
            self.values[row][col] = value
        else:
            raise IndexError("Index out of range")
    
    def determinant(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("dimensions n and m are not equal")
        if len(self.values) == 1:
            return self.values[0][0]
        if len(self.values) == 2:
            return self.values[0][0]*self.values[1][1] - self.values[0][1]*self.values[1][0]
        else:
            det = 0
            for c in tqdm(range(len(self.values))):
                A_minor = Matrix([row[:c] + row[c+1:] for row in self.values[1:]])
                det += ((-1)**c)*self.values[0][c]*A_minor.determinant()
            return det
    
    def inv(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("dimensions n and m are not equal")
        if self.determinant() == 0:
            raise ValueError("Matrix is not invertible")
        else:
            n = self.shape[0]
            if n == 1:
                return Matrix([[1/self.values[0][0]]])
            else:
                det = self.determinant()
                if det == 0:
                    raise ValueError("Matrix is not invertible")
                else:
                    result = [[0 for _ in range(n)] for _ in range(n)]
                    for i in range(n):
                        for j in range(n):
                            minor = [row[:j] + row[j+1:] for row in (self.values[:i] + self.values[i+1:])]
                            minor = Matrix(minor)
                            result[j][i] = ((-1)**(i+j))*minor.determinant()/det
                            
        return Matrix(result).transpose()
    
    # Transform a Matrix with str into integer in the columns c
    def transform_int(self, c):
        n, m = self.shape
        res = []
        for i in range(n):
            temp = []
            for j in range(m):
                if j in c:
                    temp.append(int(self.values[i][j]))
                else:
                    temp.append(self.values[i][j])
            res.append(temp)
        
        return Matrix(res)
    
    # Group the values that are from the same day
    def group_by_day(self, day_list):
        n, m = self.shape
        res = []
        for day in day_list:
            temp = Matrix([0 for _ in range(m-2)])
            for i in range(n):
                if self.values[i][0] == day:
                    temp = temp + Matrix([self.values[i][j] for j in range(2,m)])
            res.append([temp[i] for i in range(m-2)])
            
        return Matrix(res)
    
# Find the unique values of a list
def unique(List):
    res = []
    for x in List:
        if x not in res:
            res.append(x)
    return res