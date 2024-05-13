
##import numpy as np
##
##def myDef(A):
##    findZero = np.argwhere(A == 0)
##    test(findZero)
##
##
##def test(n, m):
##    np.delete(A, n, axis = 0)
##    np.delete(A, m, axis = 1)
##
##    if(A.shape >= (2, 2)):
##        myDef(A)
##    else:
##        print(A)
##
##        
##A = np.array([[1,0,-3,4], [2,-2,2,2], [-3, 2, 3, 4], [4, 2,4,-4]])
##
##print(myDef(A))
##print(np.linalg.det(A))

import numpy as np

def Mydet(matrix): 
    
    ret = 0 #answer
    dim = len(matrix) #dimension

    #escape
    if dim == 1: 
        
        ret = matrix[0][0]
        return ret
    
    else : 
        #indexing 0
        r=0 
        m=0
        for row in range(dim) :
            c = np.size(matrix[row])-np.count_nonzero(matrix[row])
            print(c)
            if c > m :
                r = row
                m = c
                
        for col in range(dim):
            a = matrix[r][col]
 
            partial = np.delete(matrix,0,axis=0)
            partial = np.delete(partial,col,axis=1)
            print(partial)
            M = Mydet(partial)
 
            sgn = (-1)**(col)
            ret += a*sgn*M
 
    return ret
A = np.array([[1,0,-3,4], [2,-2,2,2], [-3, 2, 3, 4], [4, 2,4,-4]])
print(Mydet(A))
print(np.linalg.det(A))
