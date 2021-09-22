#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as pp

#%% Mesh
N=100
x_start, x_end = -10.0, 10.0
y_start, y_end = -10.0, 10.0
x=np.linspace(x_start, x_end,N)
y=np.linspace(y_start, y_end,N)
X,Y=np.meshgrid(x,y)

E1_x = np.zeros([N,N])
E1_y = np.zeros([N,N])
E2_x = np.zeros([N,N])
E2_y = np.zeros([N,N])

a = 3.5
s = 5
e1 = 8.854*(10**(-12))
e2 = 8.854*(10**(-9))
e = (a**2)/(e1 + e2)
h = (a*s)/e2
Pox = 3
Poy = 1

#%%
F = lambda x, y: np.arctan(y/x)
R = lambda x, y: x**2 + y**2
 
E1_x = E1_x + Pox/e
E1_y = E1_y + Poy/e
E2_x = E2_x + e*Pox/R(X,Y)*(2*(np.sin(F(X,Y))**2)-1) - e*Poy/R(X,Y)*np.sin(2*F(X,Y)) + h*np.cos(F(X,Y))/np.sqrt(R(X,Y))
E2_y = E2_y + e*Poy/R(X,Y) + h*np.sin(F(X,Y))/np.sqrt(R(X,Y))

for i in range(N):
    for j in range(N):
        p = X[i,j]
        q = Y[i,j]
        w = p**2 + q**2
        if w < (a**2):
            E2_x [i,j] = 0
            E2_y [i,j] = 0
        else:
            E1_x [i,j] = 0
            E1_y [i,j] = 0

E_x = E1_x + E2_x
E_y = E1_y + E2_y

#%% Streamlines Plot
width = 15.0
height = (y_end - y_start) / (x_end - x_start) * width
pp.figure(figsize=(width, height))
pp.xlabel('x', fontsize=16)
pp.ylabel('y', fontsize=16)
pp.xlim(x_start, x_end)
pp.ylim(y_start, y_end)
pp.streamplot(X, Y, E_x, E_y,
                   density=2.0, linewidth=2, arrowsize=2, arrowstyle='->')


# In[ ]:




