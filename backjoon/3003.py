K, Q, R, B, N, P = map(int, input().split())

##king
if (K==0):
    print(K+1, end=" ")
elif(K==1):
    print(K-1, end =" ")
elif(K>=2):
    print(-(K-1), end = " ")
    
##queen
if(Q==0):
    print(Q+1, end=" ")
elif(Q==1):
    print(Q-1, end=" ")
elif(Q>=2):
    print(-(Q-1), end = " ")
    
##rook
if(R==0):
    print(R+2, end=" ")
elif(R==1):
    print(R, end=" ")
elif(R==2):
    print(R-2, end=" ")
elif(R>=3):
    print(-(R-2), end = " ")
    
#bishop
if(B==0):
    print(B+2, end=" ")
elif(B==1):
    print(B, end=" ")
elif(B==2):
    print(B-2, end=" ")
elif(B>=3):
    print(-(B-2), end = " ")

##knight
if(N==0):
    print(N+2, end=" ")
elif(N==1):
    print(N, end=" ")
elif(N==2):
    print(N-2, end=" ")
elif(N>=3):
    print(-(N-2), end = " ")
    
##phone
if(P==0):
    print(P+8, end= " ")
elif(P==1):
    print(P+6, end= " ")
elif(P==2):
    print(P+4, end= " ")
elif(P==3):
    print(P+2, end= " ")
elif(P==4):
    print(P, end= " ")
elif(P==5):
    print(P-2, end= " ")
elif(P==6):
    print(P-4, end= " ")
elif(P==7):
    print(P-6, end= " ")
elif(P==8):
    print(P-8, end = " ")
elif(P>=9):
    print(-(P-8), end=" ")

