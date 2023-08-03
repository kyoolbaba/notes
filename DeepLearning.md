
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

```
A=np.random.randn(3,4)
B=np.random.randn(4,5)
C=np.random.randn(3,7)

print(np.round(A@B,2)) , print(" ")
print(np.matmul(A,B)), print(" ")
```

![image](https://github.com/kyoolbaba/notes/assets/46890041/72d14d13-0c08-47f6-a90f-501ebf1f48d5)

```
A=torch.randn(3,4)
B=torch.randn(4,5)
C1=np.random.randn(4,7)
C2=torch.tensor(C1,dtype=torch.float)
print(A@B)
```

![image](https://github.com/kyoolbaba/notes/assets/46890041/898adf4d-d89b-46c9-b2af-3435d8692c32)


# Video 9

# Softmax Function
What is e in Math?
e=2.718
e to the power anything even negative number it results will never be negative
Because we can generate probabilities from this as probabilities need to be a positive number
Softmax of a vector is 
z=[1,2,3]
e^z={2.72,7.39,20.01}
Sum(e^z)=30.19
softmax=e^z(i)/sum(e^z)
This will result in [.09,.24,.67]
and sum of all the numbers is 1

```
a=np.array([2,6,5])
b=np.exp(a)
print(b)
c=np.sum(b)
print(c)
print(np.sum(b/c))
```
![image](https://github.com/kyoolbaba/notes/assets/46890041/ca9f4856-ae68-48e3-a0c4-2e8b7cd3df20)


![image](https://github.com/kyoolbaba/notes/assets/46890041/b342c73f-5201-4d47-9c7e-5a15f5aba71b)

![image](https://github.com/kyoolbaba/notes/assets/46890041/da90bc22-0d46-4cad-b611-fef966a3f946)

```
z=np.random.randint(5,32,100)
print(z)
num=np.exp(z)
den=np.sum(num)

sigma=num/den
print(sigma)
plt.plot(z,sigma,'bo')
plt.xlabel("Original number (z)")
plt.ylabel("Softmax transformed number")
plt.show()
plt.plot(z,sigma,'bo')
plt.xlabel("Original number (z)")
plt.yscale('log')
plt.ylabel("Softmax transformed number")
plt.show()
```

1. Below is the conversion random numbers from 5 to 35 into probabilities
![image](https://github.com/kyoolbaba/notes/assets/46890041/9b022b04-1174-400b-9f9e-04b864d57f34)
2. below is same as above but in logarithmic scale
 ![image](https://github.com/kyoolbaba/notes/assets/46890041/ff7e2dcf-42dd-420f-a04f-c93b5250e9ed)
the softmax function for normal scale is non linear but it is linear for logarithmic scale

## NOTE : exponential is an inverse of logarithm and vice versa


# Logarithmic 

![image](https://github.com/kyoolbaba/notes/assets/46890041/d082c60f-909a-4e6c-a572-55ef3aeac1af)




```

softfun=nn.Softmax()
softmax_values=softfun(torch.Tensor(np.random.randint(5,100,60)))
print(softmax_values)
torch.sum(softmax_values)
```


# Video 10 LOG
1. Log is inverse of exp and vice versa
2. Log is monotonic function of x i.e. When x goes up even y goes up and when x goes down even y goes down
3. Sin wave is non monotonic
4. Log stretches small values at the x axis
5. This is important because log distinguishes between small and closely spaced numbers

![image](https://github.com/kyoolbaba/notes/assets/46890041/76d02c3b-4650-4e63-89ba-4e5b9f6ccc89)

6. Most of Machine Learning & Deep Learning involves minimizing small quantities like probabilities.
7. Computers suffer from small numbers when working on a very small numbers
8. So the ideas is to stretch the small numbers and just makes the optimization of models to work better
Figure : This figure is an illustration for Log vs Exp(transformed to probabilities)
![image](https://github.com/kyoolbaba/notes/assets/46890041/f0958098-1506-4b7b-8979-ae64b1827481)
code for the above figure
```z=np.random.randint(5,160,100)
print(z)
num=np.exp(z)
den=np.sum(num)
sigma=num/den
log_=np.log(z)
den=np.sum(num)
plt.plot(z,sigma,'bo',label="exp")
plt.plot(z,log_,'go',label="log")
plt.legend()
plt.show()
```

















    
  
 
