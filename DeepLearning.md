
Video 6
 ## This is just teaching Vector and Matrix Transpose
- For Numpy the Code is
  ```
  import numpy as np

  nv=np.array([[1,2,3]])
  print(nv),print(" ")

  print(nv.T), print(" ")

  nvT=nv.T
  print(nv.T)

  nv=np.array([[1,2,3],[4,5,6])
  print(nv),print(" ")

  print(nv.T), print(" ")

  nvT=nv.T
  print(nv.T)
  ```
![image](https://github.com/kyoolbaba/notes/assets/46890041/d42b8ede-9eb8-468f-a6ab-32948cba380c)

  - This is a numpy code for matrix transpose

# Lets right the same code in pytorch
```
import torch
tv=torch.tensor([[1,2,3,4]])
print(tv),print(" ")
print(tv.T), print(" ")
tvT=tv.T
print(tvT),print(" ")
```
![image](https://github.com/kyoolbaba/notes/assets/46890041/9ff680dd-0e36-4fcb-a796-6dd041a1a91b)



# Video 7

## WhaT is dot product
 1. Dot product is written in angled brackets
 2. a.b <a.b> aTb
a=[1, 2, 3, 4]
b=[     [5],
        [6],
        [7],
        [8]]

a.b= (1*5)+(2*6)+(3*7)+(4*8)=70
#### the dot product of any vectors will result in a scalar value 
a. Dot product can happen between two vectors which have a specific size i.e. they should have exactly same size

Dot product can happen between on any dimensions like 2D 3D or 4 Dimensional
```
nv1=np.array([1,2,5,7])
nv2=np.array([9,3,7,1])
print(np.dot(nv1,nv2.T))
print(np.sum(nv1*nv2))
```
![image](https://github.com/kyoolbaba/notes/assets/46890041/04a1ca9a-1df1-4f8c-bb46-693cbb241b2d)

```
nv1=torch.tensor([1,2,5,7])
nv2=torch.tensor([9,3,7,1])
print(torch.dot(nv1,nv2.T))
print(torch.sum(nv1*nv2))
```



![image](https://github.com/kyoolbaba/notes/assets/46890041/da8f64a7-9016-466f-a9cf-02d5d764bfb2)

# Dot Product is a singel number that reflects commonalities between two objects (can be vector, matrices, tensors , images , signals)
## Application of dot product
1. Statistics Correlation, least-squares, entropy,PCA.
2. Signal Processing: Fourier transform, filtering
3. Science: Geometry, Physics, mechanics
4. Deep Learning: CNN,matrix multiplication, Gram Matrix(used in style transfer)

# Video 8 
# Matrix Multiplication
It is a fancy application of a dot product.
Rules for a matrix multiplication to happen
1. if a size of matrix is MxN then M is rows and N is columns, it can be only multiplied to an another matrix which is having a size of NxK
and the result will be in the size MxK
For Ex if A is a matrix of size 2x8 and B is a matrix of size 8*3 then we can multiply two matrices and the result will be in the size of  2x3





    
  
 
