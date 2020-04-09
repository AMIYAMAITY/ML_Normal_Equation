import numpy as np 
'''
Example: 
        1 x 5 = 5
        2 x 5 = 10
        3 x 5 = 15
        4 x 5 = 20
        5 x 5 = 25
        6 x 5 = 30
        ...
        ...
        ...
        10 x 5 = ? 
    Try this...
'''
# Training examples
X=[
    [1,1],
    [1,2],
    [1,3],
    [1,4],
    [1,5],
    [1,6]
]

# Output dataset
Y=[5,10,15,20,25,30]

#Formula
'''
theta= (X'.X)^-1 . (X'.Y)

[N.B]: ( ' means transpose of a matrix  and ^-1 means inverse of a matrix)

'''

x = np.array(X) 
y = np.array(Y) 

# X matrix transposing
xt=x.transpose()

# Matrix maltiplication (X'.X) and assigning to A
A=np.matmul(xt, x)
# now inversing A matrix and assigning to Ainv
Ainv=np.linalg.inv(A)

#this is actually (X'.Y) matrix multiplication
B=np.matmul(xt, y)
# This is full equation multiplication  theta= (X'.X)^-1 . (X'.Y)
theta=np.matmul(Ainv,B)

#user input line 
input_value=int(input("Enter value:"))
#calculating result according to predicting data
result=input_value * theta[1]
print("Result: {}".format(result))