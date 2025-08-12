# -*- coding: utf-8 -*-
"""
======================================
Optimal Transport for 1D distributions
======================================

This example illustrates the computation of EMD and Sinkhorn transport plans
and their visualization.

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import shiliangliang_project.sinkhorn as sll

##############################################################################
# Generate data
# -------------


#%% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=10, s=15)  # m= mean, s= std
b = gauss(n, m=55, s=8)
c = gauss(n, m=75, s=3)
e = gauss(n, m=90, s=5)
g = gauss(n, m=5, s=50)
d = (b + c + g) / np.sum(b+c+g)
f = (a + e) / np.sum(a+e)
print("a: ", np.sum(f))
print("d: ", np.sum(d))

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()


##############################################################################
# Plot distributions and loss matrix
# ----------------------------------

#%% plot the distributions

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, f, 'b', label='Source distribution')
pl.plot(x, d, 'r', label='Target distribution')
pl.legend()

#%% plot distributions and loss matrix

pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(f, d, M, 'Cost matrix M')

##############################################################################
# Solve EMD
# ---------


#%% EMD

# use fast 1D solver
G0 = ot.emd_1d(x, x, f, d)

# Equivalent to
# G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(f, d, G0, 'OT matrix G0')

##############################################################################
# Solve Sinkhorn
# --------------


#%% Sinkhorn

lambd = 1e-3
Gs = ot.sinkhorn(f, d, M, lambd)
print("Gs.shape: ", Gs.shape)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(f, d, Gs, 'OT matrix Sinkhorn')

Gsll = sll.sinkhorn_gamma(f, b / 2, 3*b/2, M, lambd)
pl.figure(5, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gsll, 'OT try gamma')
'''
print("Gsll.shape: ", Gsll.shape)
print("Gsll.sum", np.sum(Gsll))

print("a.sum", np.sum(a))
print("b.sum", np.sum(b))
print(np.sum(Gsll, axis=1))
print(np.sum(Gsll, axis=0))
print("a: ")
print(a)
print("b: ")
print(b)
'''
down, up = 0.7, 1.3
reg = [1, 0.1, 0.01, 0.005, 0.001, 0.0005]
eps = 1e-3

plt.figure(figsize=(15, 4), dpi=80)  # 创建一个10 x 10的画布，像素值为80
plt.subplot(2, 4, 1)  # 划分为2 x 1的图形阵，选择第一张图片
plt.plot(x, f, c='r')
plt.fill_between(x, 0, f, facecolor='red', alpha=0.3)
plt.legend("α")
# plt.axes([0.1, 0.1, 0.8, 0.4])
plt.subplot(2, 4, 5)
plt.plot(x, down*d, c='b')
plt.plot(x, up*d, c='g')
plt.fill_between(x, down*d, up*d, facecolor='gray', alpha=0.5)
plt.fill_between(x, 0, down*d, facecolor='blue', alpha=0.2)
plt.legend(["β$^d$", "β$^u$"])
# for i in range(6):
#     print("///////////////////////////////////////")
#     Gs = ot.sinkhorn(a, d, M, reg[i])
#     Ggamma = sll.sinkhorn_gamma(a, down*d, up*d, M, reg[i])
#     Guv = sll.sinkhorn_uv(a, down*d, up*d, M, reg[i])
#     print(Guv)
#     print("*************************************")
#     Glog = sll.sinkhorn_log(a, down*d, up*d, M, reg[i])
#     print(Glog)
#     pic.add_subplot(5, 6, 7 + i)
#     plt.imshow(Gs, cmap='Greens')
#     pic.add_subplot(5, 6, 13 + i)
#     plt.imshow(Ggamma, cmap='Reds')
#     pic.add_subplot(5, 6, 19 + i)
#     plt.imshow(Guv, cmap='Blues')
#     pic.add_subplot(5, 6, 25 + i)
#     plt.imshow(Glog, cmap='Oranges')
#     plt.xlabel(reg[i])

plt.subplot(1, 4, 2)
Ggamma = sll.sinkhorn_gamma(f, down*d, up*d, M, eps)
plt.imshow(Gs, cmap='Greens')
plt.title('Bregman')
plt.subplot(1, 4, 3)
Guv = sll.sinkhorn_uv(f, down*d, up*d, M, eps)
plt.imshow(Guv, cmap='Reds')
plt.title('Sinkhorn_Knopp')
plt.subplot(1, 4, 4)
Glog = sll.sinkhorn_log(f, down*d, up*d, M, eps)
plt.imshow(Guv, cmap='Blues')
plt.title('Dual')


# picture of title
target1 = np.sum(Gs, axis=0)
target2 = np.sum(Ggamma, axis=0)

pl.figure(7, figsize=(6.4, 3))
pl.plot(x, target1, 'b', label='Source distribution')
pl.plot(x, target2, 'r', label='Target distribution')
pl.legend()

plt.figure(8, figsize=(6.4, 9))
plt.subplot(311)
plt.plot(x, f, c='r', label='source distribution')
plt.fill_between(x, 0, f, facecolor='r', alpha=0.3)
plt.axis('off')
plt.subplot(312)
plt.plot(x, target1, c='b', label='result of sinkhorn distribution')
plt.fill_between(x, 0, target1, facecolor='b', alpha=0.3)
plt.axis('off')
plt.subplot(313)
plt.plot(x, target2, c='b', label='result of gamma distribution')
plt.plot(x, down*d, c='green', label="β$^d$", linestyle='--')
plt.plot(x, up*d, c='green', label="β$^u$", linestyle='-.')
plt.fill_between(x, 0, target2, facecolor='b', alpha=0.3)
plt.axis('off')

plt.show()
