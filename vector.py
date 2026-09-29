import numpy as np 

#Create numpy ndarray
x = np.array([3.,4.,0.])
y =  np.array([1.,2.,2.])

print(x+y)

def scalaire(a,b):
    total = 0.
    for ai,bi in zip(a,b):
        total+=ai*bi
    return total