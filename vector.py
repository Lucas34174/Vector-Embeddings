import numpy as np 
import time 
#Create numpy ndarray
x = np.array([3.,4.,0.])
y =  np.array([1,2.,2.])

print(x+y)

def dot_product(a,b):
    total = 0.
    for ai,bi in zip(a,b):
        total+=ai*bi
    return total

start = time.time()
scalaire1= dot_product(x,y)
end = time.time()
print(f"result : {scalaire1}({(end-start)*1000:.3f})")
#result : 11.0(1.4066696166992188e-05)
start = time.time()
scalaire2= x @ y 
end = time.time()
print(f"result : {scalaire1}({end-start})")
#result : 11.0(3.0517578125e-05)

#Norme vectorielle
print("Norme Vectorielle")
print(x)
print(np.sqrt(x@x))
print(np.linalg.norm(x))
print(np.linalg.norm(x,ord=1))
print(np.linalg.norm(x,ord=np.inf))

cos  = (x @ y)/( np.linalg.norm(x)*np.linalg.norm(y))
print(f"Similarité cosinus : {cos} - {np.arccos(cos)}")






