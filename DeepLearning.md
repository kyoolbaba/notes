
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
## 
    
  
 
