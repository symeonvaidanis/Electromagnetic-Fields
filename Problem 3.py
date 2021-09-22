#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as pp
#%% Mesh
N=100
x_start, x_end = -20.0, 20.0
y_start, y_end = 0, 40.0
x=np.linspace(x_start, x_end,N)
y=np.linspace(y_start, y_end,N)
X,Y=np.meshgrid(x,y)

ga = 1000
gb = 500
d = 10
v = 5

Ja_x = np.zeros([N,N])
Ja_y = (v/d)*np.ones([N,N])
Jb_x = np.zeros([N,N])
Jb_y = (-2*v/d)*np.ones([N,N])
#%%
A = lambda n: gb*(2*v*(1+2*(-1)**n)/(n*np.pi))/(ga+gb)
B = lambda n: -ga*(2*v*(1+2*(-1)**n)/(n*np.pi))/(ga+gb)

for n in range(1,100):
    
    Ja_x = Ja_x - ga*(n*np.pi/d)*A(n)*np.exp(n*np.pi*X/d)*np.sin(n*np.pi*Y/d)
    Ja_y = Ja_y - ga*(n*np.pi/d)*A(n)*np.exp(n*np.pi*X/d)*np.cos(n*np.pi*Y/d)
    
    Jb_x = Jb_x + gb*(n*np.pi/d)*B(n)*np.exp(-n*np.pi*X/d)*np.sin(n*np.pi*Y/d)
    Jb_y = Jb_y - gb*(n*np.pi/d)*B(n)*np.exp(-n*np.pi*X/d)*np.cos(n*np.pi*Y/d)

for i in range(N):
    for j in range(N):
        if X[i,j] < 0:
            Jb_x [i,j] = 0
            Jb_y [i,j] = 0
        else:
            Ja_x [i,j] = 0
            Ja_y [i,j] = 0

J_x = Ja_x + Jb_x
J_y = Ja_y + Jb_y
#%% Streamlines Plot
width = 15.0
height = (y_end - y_start) / (x_end - x_start) * width
pp.figure(figsize=(width, height))
pp.xlabel('x', fontsize=16)
pp.ylabel('y', fontsize=16)
pp.xlim(x_start, x_end)
pp.ylim(y_start, y_end)
pp.streamplot(X, Y, J_x, J_y,
                   density=2.0, linewidth=2, arrowsize=2, arrowstyle='->')


# In[ ]:




