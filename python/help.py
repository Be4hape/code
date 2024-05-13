import numpy as np

def myDet(A): #1행전개로 구현
    
    matrix=A.copy()
    det = 0 #결과를 담을 객체
    dim = len(matrix)

    if dim == 1:
        
        det = matrix[0][0] 
        return det
    
    else :
        r = 0
        m = 0
        for c in range(dim):
            j = np.size(matrix[c]) - np.count_nonzero(matrix[c])
            print(j)
            if(j > m):
                r = c
                m = j

        for col in range(dim):
            
            a = matrix[r][col]
            #소행렬식 M 추출

            partial = np.delete(matrix,0,axis=0) #행전개
            partial = np.delete(partial,col,axis=1)
            print(partial)
            M = myDet(partial) #재귀함수
 
            sgn = (-1)**(col)
            # 부호(실제 파이썬 인덱스는 0부터 시작하므로,
            #1행과 k열의 숫자 합은 col 변수와 동일한 홀/짝의 성질을 가짐)
            det += a*sgn*M
            
 
    return det
A = np.array([[1,0,-3,4], [2,-2,2,2], [-3, 2, 3, 4], [4, 2,4,-4]])
print(myDet(A))
