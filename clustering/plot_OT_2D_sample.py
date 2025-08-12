# -*- coding: utf-8 -*-

# sphinx_gallery_thumbnail_number = 4
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import clustering.sinkhorn as sll

##############################################################################
# Generate data
# -------------

# %% parameters and data generation

n = 50  # nb samples
np.random.seed(0)

mu_s_1 = np.array([0, 0])
cov_s_1 = np.array([[1, 0], [0, 1]])

mu_s_2 = np.array([5, 5])
cov_s_2 = np.array([[1, -.8], [-.8, 1]])

mu_s_3 = np.array([10, 10])
cov_s_3 = np.array([[1, -0.7], [-0.7, 1]])

mu_s_4 = np.array([15, 15])
cov_s_4 = np.array([[1, -0.88], [-0.88, 1]])

mu_s_5 = np.array([20, 20])
cov_s_5 = np.array([[1, -0.83], [-0.83, 1]])

xs_1 = ot.datasets.make_2D_samples_gauss(10, mu_s_1, cov_s_1)
print(xs_1[0])
xs_2 = ot.datasets.make_2D_samples_gauss(15, mu_s_2, cov_s_2)
xs_3 = ot.datasets.make_2D_samples_gauss(20, mu_s_3, cov_s_3)
xs_4 = ot.datasets.make_2D_samples_gauss(25, mu_s_4, cov_s_4)
xs_5 = ot.datasets.make_2D_samples_gauss(30, mu_s_5, cov_s_5)

classes = 5
a, b = np.ones((100,)) / 100, np.ones((classes,)) / classes  # uniform distribution on samples
xs = np.vstack((xs_1, xs_2, xs_3, xs_4, xs_5))
xs = np.array(xs.real)
# index = np.random.permutation(xs.shape[0])
# xs = xs[index]
# mu = np.zeros((5, 2))
# for i in range(5):
#     rand_mu_idx = np.random.randint(0, 100, 1)
#     mean = xs[rand_mu_idx]
#     mean = np.sum(mean, axis=0) / 1
#     mu[i] = mean
# print("mu: ", mu)
# mu = np.array(mu)

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
    for i in range(100):
        for j in range(classes):
            mu_new[j] += G[i, j] * x[i]

    return mu_new


# mu = np.vstack((mu_s_1, mu_s_2, mu_s_3, mu_s_4, mu_s_5))
print("xs: ********************")
print(xs)
print(xs.shape)
print("mu: ********************")
print(mu)
print(mu.shape)
C = ot.dist(xs, mu)
print("C: *********************")
print(C)
print(C.shape)
C = C / np.max(C)
# print("C: ", C)
lambd = 1e-3
numIter = 10
print("mu.shape", np.array(mu).shape)

print("*******************EMD*******************")
C_emd = C
mu_emd = mu
count = 0
while True:
    if count > numIter:
        break
    prev_mu_emd = mu_emd
    G0 = ot.emd(a, b, C_emd)
    # G0 = np.where(G0 < 1e-10, 0, G0)
    # print("*********G0*********")
    # print(G0)
    emd_idx = np.argmax(G0, axis=1)
    # mu_emd = np.array(compute_mu_new(xs, G0))
    mu_emd = np.array(compute_mu(xs, emd_idx, classes))
    # if np.sum(np.abs(mu_emd - prev_mu_emd)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_emd = compute_C(xs, mu_emd)
    C_emd = ot.dist(xs, mu_emd)
    C_emd = C_emd / np.max(C_emd)

print("*******************SINKHORN*******************")
C_sink = C
mu_sink = mu
count = 0
while True:
    if count > numIter:
        break
    prev_mu_sink = mu_sink
    Gs = ot.sinkhorn(a, b, C_sink, lambd, numItermax=1000)
    # Gs = np.where(Gs < 1e-10, 0, Gs)
    # print("******Gs*******")
    # print(Gs)
    sinkhorn_idx = np.argmax(Gs, axis=1)
    # mu_sink = np.array(compute_mu_new(xs, Gs))
    mu_sink = np.array(compute_mu(xs, sinkhorn_idx, classes))
    # if np.sum(np.abs(mu_sink - prev_mu_sink)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sink = compute_C(xs, mu_sink)
    C_sink = ot.dist(xs, mu_sink)
    C_sink = C_sink / np.max(C_sink)

print("*******************SLL*******************")
C_sll = C
mu_sll = mu
count = 0
while True:
    if count > numIter:
        break
    prev_mu_sll = mu_sll
    Gsll = sll.sinkhorn_gamma(a, b / 2, 3 * b / 2, C_sll, lambd)
    # Gsll = np.where(Gsll < 1e-10, 0, Gsll)
    # print("*************Gsll**************")
    # print(Gsll)
    sll_idx = np.argmax(Gsll, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_sll = np.array(compute_mu(xs, sll_idx, classes))
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_sll = ot.dist(xs, mu_sll)
    C_sll = C_sll / np.max(C_sll)

print("*****************LAGRANGE*****************")
C_lag = C
mu_lag = mu
count = 0
while True:
    if count > numIter:
        break
    prev_mu_lag = mu_lag
    Glag = sll.sinkhorn_uv(a, b / 2, 3 * b / 2, C_lag, lambd)
    # Gsll = np.where(Gsll < 1e-10, 0, Gsll)
    # print("*************Gsll**************")
    # print(Gsll)
    lag_idx = np.argmax(Glag, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_lag = np.array(compute_mu(xs, lag_idx, classes))
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_lag = ot.dist(xs, mu_lag)
    C_lag = C_lag / np.max(C_lag)

print("*****************LOG*****************")
C_log = C
C_log1 = C
mu_log = mu
mu_log1 = mu
count = 0
while True:
    if count > numIter:
        break
    prev_mu_log = mu_log
    Glog = sll.sinkhorn_log(a, b / 2, 3 * b / 2, C_log, lambd)
    Glog1 = sll.sinkhorn_log(a, b / 6, 3 * b / 2, C_log1, lambd)
    # Gsll = np.where(Gsll < 1e-10, 0, Gsll)
    # print("*************Gsll**************")
    # print(Gsll)
    log_idx = np.argmax(Glog, axis=1)
    log_idx1 = np.argmax(Glog1, axis=1)
    # mu_sll = np.array(compute_mu_new(xs, Gsll))
    mu_log = np.array(compute_mu(xs, log_idx, classes))
    mu_log1 = np.array(compute_mu(xs, log_idx1, classes))
    # if np.sum(np.abs(mu_sll - prev_mu_sll)) < 1e-70:
    #     print("num of iter:", count)
    #     break
    count += 1
    # C_sll = compute_C(xs, mu_sll)
    C_log = ot.dist(xs, mu_log)
    C_log = C_log / np.max(C_log)
    C_log1 = ot.dist(xs, mu_log1)
    C_log1 = C_log1 / np.max(C_log1)

##############################################################################
# Plot data
# ---------

# %% plot samples

# xs = xs[np.argsort(index)]
pl.figure(1)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu[:, 0], mu[:, 1], 'xr', label='Center samples')
pl.legend(loc=0)
pl.title('Source and target distributions')

pl.figure(2)
pl.imshow(C, interpolation='nearest')
pl.title('Cost matrix M')
# pl.show()

##############################################################################
# Compute EMD
# -----------

# %% EMD

# G0 = ot.emd(a, b, C)
# print("G0: ", G0)
emd_idx = np.argmax(G0, axis=1)
print("ot.emd.idx: ", emd_idx)
print("emd列和：", np.sum(G0, axis=0))

# pl.figure(3)
# pl.imshow(G0, interpolation='nearest')
# pl.title('OT matrix G0')

pl.figure(4)
# ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_emd[:, 0], mu_emd[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix emd')
print(f'Emd mu: {mu_emd}')


##############################################################################
# Compute Sinkhorn
# ----------------

# %% sinkhorn

# reg term
# lambd = 1e-5

# Gs = ot.sinkhorn(a, b, C, lambd, numItermax=1000)
# print("Gs: ", Gs)
sinkhorn_idx = np.argmax(Gs, axis=1)
print("ot.sinkhorn.idx: ", sinkhorn_idx)
print("sinkhorn列和：", np.sum(Gs, axis=0))

# pl.figure(5)
# pl.imshow(Gs, interpolation='nearest')
# pl.title('OT matrix sinkhorn')

pl.figure(6)
# ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_sink[:, 0], mu_sink[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix sinkhorn')
print(f'sinkhorn mu: {mu_sink}')


##############################################################################
# Emprirical Sinkhorn
# -------------------

# %% sinkhorn

# reg term
# lambd = 1e-1
#
# Ges = ot.bregman.empirical_sinkhorn(xs, xt, lambd)
#
# pl.figure(7)
# pl.imshow(Ges, interpolation='nearest')
# pl.title('OT matrix empirical sinkhorn')
#
# pl.figure(8)
# ot.plot.plot2D_samples_mat(xs, xt, Ges, color=[.5, .5, 1])
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# pl.title('OT matrix Sinkhorn from samples')

# Gsll = sll.sinkhorn_gamma(a, b, C, 1e-4)
# print("Gsll: ", Gsll)
sll_idx = np.argmax(Gsll, axis=1)
print("sll.idx: ", sll_idx)
print("sll列和：", np.sum(Gsll, axis=0))

# pl.figure(9)
# pl.imshow(Gsll, interpolation='nearest')
# pl.title('OT matrix try')

pl.figure(10)
# ot.plot.plot2D_samples_mat(xs, xt, Gsll, color=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_sll[:, 0], mu_sll[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix ours')
print(f'ours mu: {mu_sll}')

lag_idx = np.argmax(Glag, axis=1)
print("slag.idx: ", lag_idx)
print("slag列和：", np.sum(Glag, axis=0))

# pl.figure(11)
# pl.imshow(Glag, interpolation='nearest')
# pl.title('OT matrix try')

pl.figure(12)
# ot.plot.plot2D_samples_mat(xs, xt, Gsll, color=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_lag[:, 0], mu_lag[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix ours lagrange')
print(f'ours mu: {mu_lag}')

log_idx = np.argmax(Glog, axis=1)
print("slog.idx: ", log_idx)
print("slog列和：", np.sum(Glog, axis=0))

# pl.figure(13)
# pl.imshow(Glog, interpolation='nearest')
# pl.title('OT matrix try')

pl.figure(14)
# ot.plot.plot2D_samples_mat(xs, xt, Gsll, color=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_log[:, 0], mu_log[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix ours log')
print(f'ours mu: {mu_log}')

log_idx1 = np.argmax(Glog1, axis=1)
print("slog1.idx: ", log_idx1)
print("slog1列和：", np.sum(Glog1, axis=0))

# pl.figure(15)
# pl.imshow(Glog1, interpolation='nearest')
# pl.title('OT matrix try')

pl.figure(16)
# ot.plot.plot2D_samples_mat(xs, xt, Gsll, color=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(mu_log1[:, 0], mu_log1[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix ours log1')
print(f'ours mu: {mu_log1}')

pl.show()
