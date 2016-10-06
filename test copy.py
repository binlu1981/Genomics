git test
numpy 

import numpy as np


print np.arange(18)
#[ 0 1 2 ... 17]
a = np.arange(18).reshape(9,2)
#[[ 0  1]
# [ 2  3]
# ...
# [16 17]]
b = a.reshape(3,3,2).swapaxes(0,2)
print b.shape
#(2,3,3)
print a
print a.reshape(3,3,2)
"""
[[[ 0  1]
  [ 2  3]
  [ 4  5]]
 [[ 6  7]
  [ 8  9]
  [10 11]]
 [[12 13]
  [14 15]
  [16 17]]]
"""
print a.reshape(2,3,3)
"""
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  
  [ 9 10 11]
  [12 13 14]
  [15 16 17]]]
 """
print b
"""
[[[ 0  6 12]
  [ 2  8 14]
  [ 4 10 16]]
 [[ 1  7 13]
  [ 3  9 15]
  [ 5 11 17]]]
"""



a = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
b = a.reshape((2, 2, 3)).swapaxes(0, 2)  
### original 3 col data clust into to 3 new groups
### row index did not change
### for each new group, 2 new col should from different 2 original groups
c = a.reshape((2, 2, 3)).reshape(2,1,6)
print a.reshape(2, 2, 3)
"""
[[[ 1  2  3]
  [ 4  5  6]]
 [[ 7  8  9]
  [10 11 12]]]
"""
print(b)
"""
[[[ 1  7]
  [ 4 10]]
 [[ 2  8]
  [ 5 11]]
 [[ 3  9]
  [ 6 12]]]
"""
print(c)
"""
[[[ 1 2 3 4 5 6]]
 [[ 7 8 9 10 11 12]]]
"""
print a.reshape((2, 2, 3)).swapaxes(0, 1)
"""
[[[ 1  2  3]
  [ 7  8  9]]
 [[ 4  5  6]
  [10 11 12]]]
"""
a = np.arange(0,100,10)
#[0 10 20 ... 90]
b = a[[1,5,-1]]
#[10 50 90]
print a
print b

a = np.arange(12).reshape(2,3,2)
print a 
"""
[[[ 0  1]
  [ 2  3]
  [ 4  5]]
 [[ 6  7]
  [ 8  9]
  [10 11]]]
"""
#how to exchange position of [4 5] and [10 11]?
b = a.swapaxes(0,1)
print b
"""
[[[ 0  1]
  [ 6  7]]
 [[ 2  3]
  [ 8  9]]
 [[ 4  5]
  [10 11]]]
"""
#how to exchange position of [4 5] and [10 11]?
print a[:,2,:]
"""
[[ 4, 5]
 [10,11]]
"""
print a[::-1,2,:]
"""
[[10,11]
 [ 4, 5]]
"""
a[:,2,:]=a[::-1,2,:]
print a 
"""
[[[ 0  1]
  [ 2  3]
  [10 11]]
 [[ 6  7]
  [ 8  9]
  [ 4  5]]]
"""
import matplotlib.pyplot as plt
a = np.linspace(0,2*np.pi,50)
b = np.sin(a)
plt.plot(a,b)
mask = b >=0
plt.plot(a[mask],b[b>=0],'bo')
mask = (b >=0) & (a<= np.pi/2)
plt.plot(a[mask],b[mask])
plt.show()

a = np.arange(0,100,10)
print a[a>60]
#[70,80,90]
print np.where(a > 60)
#(array([7,8,9]),)
print np.where(a>60)[0]
#[7,8,9]
