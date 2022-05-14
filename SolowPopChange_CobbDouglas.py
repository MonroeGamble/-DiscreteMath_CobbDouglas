# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:20:43 2021

@author: jmg2g
"""

# Import Librarires 
import numpy as np
import matplotlib.pyplot as plt



Tf  = 10
T = 5

dt = 0.01 # time step
s_k = .2  # Savings rate (capial)
s_h = .2
g = 0.01  # rate of growth for TFP
delta = 0.03 # depreciation
alpha = .33 #fraction of income appropriated by capital
beta = .33

n = .05 # rate of pop. growth (before)
nprime = .15 #.0536 # rate after shock




## Take the current pop. growth (n)
## LooK rate of increas in pop. if we relese people
### For Example .... Increase nprime % release * jailed population (20,000)

## 1 million people
## 



# Define Array of T
t = np.arange(0, Tf+dt, dt)


Lbef = 1*np.ones(t.shape)
Laf = 1*np.ones(t.shape)
Abef = 1*np.ones(t.shape)
Aaf = 1*np.ones(t.shape)
Ybef = 1*np.ones(t.shape)
Yaf = 1*np.ones(t.shape)
Kbef = 1*np.ones(t.shape)
Kaf = 1*np.ones(t.shape)
Hbef = 1*np.ones(t.shape)
Haf = 1*np.ones(t.shape)


Lbef[0] = 1
Laf[0] = 1
Abef[0] = 1
Aaf[0] = 1
Kbef[0]  = 1
Kaf[0] = 1
Hbef[0] = 1
Haf[0] = 1
Ybef[0] = (Abef[0]*Lbef[0])**alpha*Kbef[0]**(1-alpha)
Yaf[0] = (Aaf[0]*Laf[0])**alpha*Kaf[0]**(1-alpha) # May need to change 

# k[0] = (Y[0]/(A[0]*L[0]))**(1/alpha)

for j in range(len(t)-1):
    Abef[j+1] = Abef[0]*np.exp(g*t[j+1])
    Aaf[j+1] = Aaf[0]*np.exp(g*t[j+1])
    
    if t[j+1] <= T:
        Lbef[j+1] = Lbef[0]*np.exp(n*t[j+1])
        Laf[j+1] = Laf[0]*np.exp(n*t[j+1])
        
    else:
        Lbef[j+1] = Lbef[0]*np.exp(n*T)*np.exp(n*(t[j+1]-T))
        Laf[j+1] = Laf[0]*np.exp(n*T)*np.exp(nprime*(t[j+1]-T))
                
        
    Kbef[j+1] = Kbef[j] + dt*(s_k*Ybef[j]-delta*Kbef[j])
    Kaf[j+1] = Kaf[j] + dt*(s_k*Yaf[j]-delta*Kaf[j])
    
    Hbef[j+1] = Hbef[j] + dt*(s_h*Ybef[j]-delta*Hbef[j])
    Haf[j+1] = Haf[j] + dt*(s_h*Yaf[j] - delta*Haf[j])
        
    Ybef[j+1] = Kbef[j+1]**alpha*Hbef[j+1]**beta*(Abef[j+1]*Lbef[j+1])**(1-alpha-beta)
    Ybef[j+1] = Kbef[j+1]**alpha*Hbef[j+1]**beta*(Abef[j+1]*Lbef[j+1])**(1-alpha-beta)
    
    (Abef[j+1]*Lbef[j+1])**alpha*Kbef[j+1]**(1-alpha)
    Yaf[j+1] = (Aaf[j+1]*Laf[j+1])**alpha*Kaf[j+1]**(1-alpha)
    
   #  k[j+1] = (Y[j+1]/(A[j+1]*L[j+1]))**(1/alpha)

# Before Change
plt.plot(t, Abef, '--', label = 'A')
plt.plot(t, Kbef, 's', label = 'K')
plt.plot(t, Lbef, 'o', label = 'L')
plt.plot(t, Ybef, '*', label = 'Y')

plt.xlabel('Time')
plt.ylabel('Capital')
plt.title('Before Change')
plt.legend()
plt.show()

# After Change
plt.plot(t, Aaf, '--', label = 'A')
plt.plot(t, Kaf, 's', label = 'K')
plt.plot(t, Laf, 'o', label = 'L')
plt.plot(t, Yaf, '*', label = 'Y')

plt.xlabel('Time')
plt.ylabel('Capital')
plt.title('After Change')
plt.legend()
plt.show()

plt.plot(t,Ybef, '--', label = 'Ybef')
plt.plot(t, Yaf, '--', label = 'Yaf')
plt.legend()
plt.show()

# plt.plot(K[0:-2], K[1:-1], '--', label = 'A')
# plt.plot(k[0,-2], k[1,-1], 'o', label = 'k')


# # Cobb Douglas
# # https://matplotlib.org/3.5.0/gallery/mplot3d/surface3d.html

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# import numpy as np

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# Percent Change in Output
YChange = ((Yaf - Ybef) / Yaf)*100
print(YChange[-1])

plt.plot(t, YChange, '--', label = 'Ybef')
plt.xlabel('Time')
plt.ylabel('% Change in Output')
plt.legend()
plt.show()

# 1) Completely Remove from working pop (full employment)
# 2) Add to working pop +6 million
# 3) Actual Data


#18000000000000 * .007 = 126,000,000,000


# unemployment of pop. vs. unemployment of person's convicted of crime