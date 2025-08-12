# -*- coding: utf-8 -*-

# sphinx_gallery_thumbnail_number = 4
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import shiliangliang_project.sinkhorn as sll
from sklearn.cluster import k_means

##############################################################################
# Generate data
# -------------

# %% parameters and data generation

n = 50  # nb samples
np.random.seed(0)

mu_s_1 = np.array([0, 5])
cov_s_1 = np.array([[1, -0.32], [-0.32, 0.91]])

mu_s_2 = np.array([4.7553, 1.5451])
cov_s_2 = np.array([[1, 0.718], [0.518, 1]])

mu_s_3 = np.array([2.9389, -4.0451])
cov_s_3 = np.array([[1, -0.938], [-0.938, 1]])

mu_s_4 = np.array([-2.9389, -4.0451])
cov_s_4 = np.array([[1, -0.69], [-0.69, 1]])

mu_s_5 = np.array([-4.7553, 1.5451])
cov_s_5 = np.array([[1, 0.466], [0.466, 1]])

xs_1 = ot.datasets.make_2D_samples_gauss(10, mu_s_1, cov_s_1)
print(xs_1[0])
xs_2 = ot.datasets.make_2D_samples_gauss(20, mu_s_2, cov_s_2)
xs_3 = ot.datasets.make_2D_samples_gauss(30, mu_s_3, cov_s_3)
xs_4 = ot.datasets.make_2D_samples_gauss(40, mu_s_4, cov_s_4)
xs_5 = ot.datasets.make_2D_samples_gauss(50, mu_s_5, cov_s_5)

classes = 5
n_point = 150
label = np.zeros(10)
label = np.hstack((label, np.full(20, 1), np.full(30, 2), np.full(40, 3), np.full(50, 4)))
print(label)
print(label.shape)

xs = np.vstack((xs_1, xs_2, xs_3, xs_4, xs_5))
xs = np.array(xs.real)

print("xs.shape: ", xs.shape)


def distance_to_nearest_centroid(x, m):
    distance = ot.dist(x, m)
    # print("distance: ", distance)
    min_dist = np.min(distance, axis=1)
    # print("min_dist: ", min_dist)
    return min_dist


# Initialize centroids list
mu = [xs[np.random.randint(0, xs.shape[0])]]
print("mu: ", mu)
# Randomly select first centroid

# Select remaining centroids
for i in range(1, classes):
    # Calculate distance of each data point to nearest centroid
    distances = distance_to_nearest_centroid(xs, np.array(mu))
    # Convert distances to probability distribution
    sum_distances = np.sum(distances)
    probabilities = distances / sum_distances
    # Randomly select a new centroid from probability distribution
    r = np.random.random()
    cumulative_probability = 0
    for j, p in enumerate(probabilities):
        cumulative_probability += p
        if r <= cumulative_probability:
            mu.append(xs[j])
            break

mu = np.array(mu)
print("mu: ", mu)


def compute_mu(x, idx, classes):
    mu_new = np.zeros((classes, 2))
    for k in range(classes):
        # print("idx: ", k)
        rank = np.where(idx == k)
        # print("rank: ", rank)
        m = x[rank]
        # print("m: ", m)
        mk = np.sum(m, axis=0) / len(rank[0])
        # print(len(rank[0]))
        mu_new[k] = mk
    return mu_new


def compute_mu_new(x, G, classes):
    mu_new = np.zeros((classes, 2))
    for i in range(x.shape[0]):
        for j in range(classes):
            mu_new[j] += G[i, j] * x[i]

    return mu_new


mu = xs[np.random.randint(0, n_point, 5)]
# mu = np.vstack((mu_s_1, mu_s_2, mu_s_3, mu_s_4, mu_s_5))
print("xs: ********************")
# print(xs)
print(xs.shape)
print("mu: ********************")
print(mu)
print(mu.shape)
C = ot.dist(xs, mu)
print("C: *********************")
# print(C)
print(C.shape)
C = C / np.max(C)
# print("C: ", C)
lambd = 1e-3
numIter = 8
down = 0.6
up = 1.3
ii = 2
a = np.ones((n_point,)) / n_point  # uniform distribution on samples
b = np.array([1, 1, 1, 3, 1])
b = b / np.sum(b)
print("mu.shape", np.array(mu).shape)

C_ot = C
C_ot1 = C
mu_ot = mu
count = 0
t0 = time.time()
list_mu_ot = []
list_Got = []
while True:
    if count > numIter:
        break
    prev_mu_ot = mu_ot
    Got = sll.sinkhorn_gamma_break(a, b*down, b*up, C_ot, lambd)
    if count % ii == 0:
        list_mu_ot.append(mu_ot)
        list_Got.append(Got)
    Got1 = sll.sinkhorn_gamma_break(a, b*down, b*up, C_ot1, lambd)
    ot_idx = np.argmax(Got1, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_ot = np.array(compute_mu_new(xs, Got, classes))
    mu_ot1 = np.array(compute_mu(xs, ot_idx, classes))
    # if np.sum(prev_mu_ot - mu_ot) < 1e-7:
    #     break
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_ot = ot.dist(xs, mu_ot)
    C_ot = C_ot / np.max(C_ot)
    C_ot1 = ot.dist(xs, mu_ot1)
    C_ot1 = C_ot1 / np.max(C_ot1)
t1 = time.time()

C_sink = C
mu_sink = mu
count = 0
list_Gsink = []
list_mu_sink = []
while True:
    if count > numIter:
        break
    prev_mu_sink = mu_sink
    Gsink = ot.sinkhorn(a, b, C_sink, lambd)
    if count % ii == 0:
        list_mu_sink.append(mu_sink)
        list_Gsink.append(Gsink)
    # ot_idx = np.argmax(Got, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_sink = np.array(compute_mu_new(xs, Gsink, classes))
    # if np.sum(prev_mu_sink - mu_sink) < 1e-7:
    #     break
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_sink = ot.dist(xs, mu_sink)
    C_sink = C_sink / np.max(C_sink)
t2 = time.time()

C_gamma = C
mu_gamma = mu
count = 0
list_Ggamma = []
list_mu_gamma = []
while True:
    if count > numIter:
        break
    prev_mu_gamma = mu_gamma
    Ggamma = sll.sinkhorn_gamma(a, b*down, b*up, C_gamma, lambd)
    if count % ii == 0:
        list_mu_gamma.append(mu_gamma)
        list_Ggamma.append(Ggamma)
    gamma_idx = np.argmax(Ggamma, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_gamma = np.array(compute_mu_new(xs, Ggamma, classes))
    # if np.sum(prev_mu_gamma - mu_gamma) < 1e-7:
    #     break
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_gamma = ot.dist(xs, mu_gamma)
    C_gamma = C_gamma / np.max(C_gamma)
t3 = time.time()

C_sll = C
mu_sll = mu
count = 0
list_Gsll = []
list_mu_sll = []
while True:
    if count > numIter:
        break
    prev_mu_sll = mu_sll
    Gsll = sll.sinkhorn_gamma(a, b*down, b*up, C_sll, lambd)
    if count % ii == 0:
        list_mu_sll.append(mu_sll)
        list_Gsll.append(Gsll)
    sll_idx = np.argmax(Gsll, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_sll = np.array(compute_mu(xs, sll_idx, classes))
    # if np.sum(prev_mu_sll - mu_sll) < 1e-7:
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_sll = ot.dist(xs, mu_sll)
    C_sll = C_sll / np.max(C_sll)
t4 = time.time()

cen, y, interia = k_means(xs, 5)

t5 = time.time()

t_lda = t1 - t0
t_mean = t3 - t2
t_class = t4 - t3
t_kmeans = t5 - t4
print("t_lda: ", t_lda)
print("t_mean: ", t_mean)
print("t_class: ", t_class)
print("t_kmeans: ", t_kmeans)

print(f'ot mu: {list_mu_ot[-1]}')
print(f"gamma mu: {list_mu_gamma[-1]}")
print(f'sll mu: {list_mu_sll[-1]}')
print(f'kmeans mu: {cen}')

'''
pl.figure(1)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu[:, 0], mu[:, 1], 'xr', label='Center samples')
pl.legend(loc=0)
pl.title('Source and target distributions')

pl.figure(2)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_ot[:, 0], mu_ot[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix OTLDA')
print(f'Ot mu: {mu_ot}')

pl.figure(3)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_sink[:, 0], mu_sink[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix sinkhorn')
print(f'sink mu: {mu_sink}')

pl.figure(4)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_gamma[:, 0], mu_gamma[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix gamma')
print(f'gamma mu: {mu_gamma}')

pl.figure(5)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_sll[:, 0], mu_sll[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix sll')
print(f'sll mu: {mu_sll}')

pl.figure(6)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_ot1[:, 0], mu_ot1[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix ot1')
print(f'ot1 mu: {mu_ot1}')

pl.show()
'''

plt.figure(figsize=(25, 18))
for i in range(int(numIter / ii) + 1):
    plt.subplot(3, 5, 1+i)
    ot_idx = np.argmax(list_Got[i], axis=1)
    plt.scatter(xs[:, 0], xs[:, 1], c=ot_idx, s=10, label='Source samples')
    plt.plot(list_mu_ot[i][:, 0], list_mu_ot[i][:, 1], 'xr', label='Center samples')

    plt.subplot(3, 5, 6+i)
    gamma_idx = np.argmax(list_Ggamma[i], axis=1)
    plt.scatter(xs[:, 0], xs[:, 1], c=gamma_idx, s=10, label='Source samples')
    plt.plot(list_mu_gamma[i][:, 0], list_mu_gamma[i][:, 1], 'xr', label='Center samples')

    plt.subplot(3, 5, 11+i)
    sll_idx = np.argmax(list_Gsll[i], axis=1)
    plt.scatter(xs[:, 0], xs[:, 1], c=sll_idx, s=10, label='Source samples')
    plt.plot(list_mu_sll[i][:, 0], list_mu_sll[i][:, 1], 'xr', label='Center samples')


plt.figure(2)
plt.scatter(xs[:, 0], xs[:, 1], c=y, s=10)
plt.plot(cen[:, 0], cen[:, 1], 'xr')

numClass = 5
idx = np.argmax(list_Gsink[-1], axis=1)
print(idx)
resulte = []
resulte_label = []
num_of_e_class = []
for i in range(numClass):
    print(f"num of class {i}:")
    print(len(np.where(idx == i)[0]))
    num_of_e_class.append(len(np.where(idx == i)[0]))
    resulte.append(np.where(idx == i)[0])
    resulte_label.append(label[np.where(idx == i)[0]])

resulte = np.array(resulte)
resulte_label = np.array(resulte_label)
print("result.shape: ", resulte.shape)
print("result_label.shape: ", resulte_label.shape)

total = np.zeros((numClass, numClass))
for i in range(numClass):
    for j in range(numClass):
        total[i, j] = len(np.where(resulte_label[i] == j)[0])

print("total:")
print(total)

acc = np.sum(np.max(total, axis=1)) / np.sum(total)
print("acc:")
print(acc)

plt.show()
