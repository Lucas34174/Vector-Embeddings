import numpy as np 
import time 
#Create numpy ndarray
x = np.array([3.,4.,0.])
y =  np.array([1.,2.,2.])

print(x+y)

def dot_product(a,b):
    total = 0.
    for ai,bi in zip(a,b):
        total+=ai*bi
    return total

start = time.time()
scalaire1= dot_product(x,y)
end = time.time()
print(f"result : {scalaire1}({end-start})")
start = time.time()
scalaire2= x @ y 
end = time.time()
print(f"result : {scalaire1}({end-start})")

