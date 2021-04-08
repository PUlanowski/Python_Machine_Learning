# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:35:11 2020

@author: pit
"""
#WYKRESY

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = np.linspace(0,10)
y = np.sin(x)

plt.plot(x,y)


x = np.linspace(0, 10, 5)
y = np.cos(x**1/3)

plt.scatter(x,y)


f1 = interpolate.interp1d(x, y, kind='linear')
f2 = interpolate.interp1d(x, y, kind='cubic')
x2 = np.linspace(0, 10, 50)

plt.figure() # create a plot figure
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));

#style
print(plt.style.available)
plt.style.use('ggplot')

#tworzenie nowej figury
x = np.linspace(0, 10)
y = np.sin(x)
y2 = np.cos(x)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y)
ax2.plot(x, y2)

#dwa wykresy na jednej osi
import math
fig, ax = plt.subplots()
ax.plot(x,y)
ax.set(xlim=[0, 2*math.pi], xlabel='x', ylabel='y',
title='sin(x)')
ax.axhline(0.5, ls='--', color='r', linewidth=3, alpha=0.2)

#plt.axis([-1, 11, -1.5, 1.5]);
#dwa wykresy na jednej figurze
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(x, y)
ax1.set_title('X axis shared')
x2 = np.linspace(-10, 0)
ax2.scatter(x2, y)


#zapisywanie figury
fig.canvas.get_supported_filetypes()
fig.savefig('sin.png', transparent=False, dpi=120, bbox_inches='tight')
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.sin(x), color='blue') #ustawienie koloru
plt.plot(x, x + 5, linestyle='--') #lina przerywana
plt.plot(x, x + 1, '-g')
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('sin(x)');
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.axis('equal')
plt.legend();

#zapisywanie figury
fig = plt.figure()
plt.scatter(x, y, marker='x');
data = np.random.randn(50)
plt.hist(data, bins=5, alpha=0.9,
histtype='stepfilled', color='orange',
edgecolor='none')
fig.savefig('D:\MachineLearningPandaIT\Materials\my_figure.png')

#scipy
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
x = np.linspace(0, 10, 5)
y = np.cos(x**1/3)
f1 = interpolate.interp1d(x, y,kind = 'linear')
f2 = interpolate.interp1d(x, y, kind = 'cubic')
x2 = np.linspace(0, 10, 50)
plt.plot(x, y, 'o', x, f1(x), ':', x2, f2(x2), '-')
plt.legend(['real data', 'linear', 'cubic'], loc = 'best')
plt.show()

