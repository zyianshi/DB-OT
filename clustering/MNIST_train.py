import time
from MNIST_loader import load
import matplotlib.pyplot as plt
import numpy as np
import ot
import sinkhorn
# from numba import cuda
# import numba as nb
import torch
from sklearn.cluster import k_means
from ot.datasets import make_1D_gauss as gauss

fig_size = 28

t = np.linspace(0, fig_size - 1, fig_size)
# print(t)
# t_n = t[::-1]
# t = np.array([t, t_n]).reshape(-1)
# print("t.shape: ", t.shape)
[Y, X] = np.meshgrid(t, t)
# print(X.reshape(-1))
# print(Y.reshape(-1))
position = np.vstack((X.reshape(-1), Y.reshape(-1)))
position = position.T / np.max(position)
print("position.shape: ", position.shape)
# print("position: ", position)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# W distance
def distance_to_centroid(x, m, D, reg):
    row, col = x.shape[0], len(m)
    print("row: ", row)
    print("col: ", col)
    distance = torch.zeros((col, row))
    Pij = torch.zeros(row)
    for i in range(col):
        Pij = sinkhorn.sinkhorn_knopp_new(torch.tensor(m[i]), torch.tensor(x.T), torch.tensor(D), reg)
        distance[i] = Pij
    # print("distance: ", distance)
    # print("min_dist: ", min_dist)
    return distance.T


# Euler distance
# def distance_to_centroid(x, m, D, reg):
#     row, col = x.shape[0], len(m)
#     print("row: ", row)
#     print("col: ", col)
#     distance = ot.dist(x, m)
#     print("distance: ", distance.shape)
#     # print("min_dist: ", min_dist)
#     return distance


def initial_center(xs, classes, Dist, reg):
    # Initialize centroids list
    # mu = torch.zeros((classes, xs.shape[1]))
    mu = [xs[np.random.randint(0, xs.shape[0])]]
    # print("mu: ", mu)
    # Randomly select first centroid

    # Select remaining centroids
    for i in range(1, classes):
        # Calculate distance of each data point to nearest centroid
        # print("************************************************")
        distances = distance_to_centroid(torch.tensor(xs), torch.tensor(np.array(mu)), torch.tensor(Dist), reg)
        # print("************************************************")
        distances = torch.min(distances, dim=1).values
        # Convert distances to probability distribution
        sum_distances = torch.sum(distances)
        probabilities = distances / sum_distances
        # Randomly select a new centroid from probability distribution
        r = torch.rand(1)
        cumulative_probability = 0
        for j, p in enumerate(probabilities):
            cumulative_probability += p
            if r <= cumulative_probability:
                mu.append(xs[j])
                break
    return np.array(mu)


def compute_mu(x, classes, D, reg, G):
    mu_new = torch.zeros((classes, x.shape[1]))
    for k in range(classes):
        mk = sinkhorn.barycenter_sinkhorn(x.T, D, reg, G.T[k])
        # print(f"G.T[{k}].shape", G.T[k].shape)
        mu_new[k] = mk.T
    # print("mu_new: ")
    # print(mu_new)

    return mu_new


def compute_mu_new(x, classes, D, reg, G):
    mu_new = torch.zeros((classes, x.shape[1]))
    idx = torch.argmax(G, dim=1)
    for k in range(classes):
        xk_sum = torch.sum(x[torch.where(idx == k)], dim=0)
        mu_new[k] = xk_sum / len(torch.where(idx == k)[0])
    # print("mu_new: ")
    # print(mu_new.shape)

    return mu_new


def to_one_hot(label, dimension=784):
    results = torch.zeros((len(label), dimension))
    for i, label in enumerate(label):
        results[i, label] = 1.
    return results


def calculate_func(C0, mu0, xs, classes, Dist, reg, numIter=5, func_type='sinkhorn'):
    print(f"*******************{func_type}*******************")
    mu = mu0
    C = torch.tensor(C0, dtype=torch.float)
    mu_s = mu0
    C_s = torch.tensor(C0, dtype=torch.float)
    count = 0
    a = torch.ones((xs.shape[0],), dtype=torch.float) / xs.shape[0]
    b = torch.ones((classes,), dtype=torch.float) / classes
    # b = torch.tensor(np.array([3, 6, 9, 12, 15, 15, 12, 9, 6, 3]))
    G = torch.zeros((xs.shape[0], classes), dtype=torch.float)
    G_s = torch.zeros((xs.shape[0], classes), dtype=torch.float)
    prev_mu = torch.zeros((classes, xs.shape[1]))
    prev_C = torch.zeros((xs.shape[0], classes))
    prev_mu_s = torch.zeros((classes, xs.shape[1]))
    prev_C_s = torch.zeros((xs.shape[0], classes))
    # tau = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
    # tau = torch.tensor(np.array(range(classes))[::-1] / classes, dtype=torch.float)
    tau = torch.tensor(1 - np.array(gauss(16, m=8, s=5)) * 10, dtype=torch.float)
    print("tau:", tau)
    G_list = []
    mu_list = []
    # for i in range(6):
    #     mu = mu0
    #     C = torch.tensor(C0, dtype=torch.float)
    #     mu_s = mu0
    #     C_s = torch.tensor(C0, dtype=torch.float)
    #     count = 0
    while True:
        if count > numIter:
            break

        G = sinkhorn.sinkhorn_gamma(a.T, b.T * (1-tau), b.T * (1+tau), C, reg)
        G_s = ot.sinkhorn(a.T, b.T, C, reg)
        # print("G:")
        # print(G)
        # print("G.T")
        # print(np.sum(G, axis=0))
        idx = torch.argmax(G, dim=1)
        idx_s = torch.argmax(G_s, dim=1)
        one_hot_G = to_one_hot(idx, classes)
        one_hot_G_s = to_one_hot(idx_s, classes)
        # print("one_hot_G:")
        # print(one_hot_G)
        G_new = G * one_hot_G
        G_s_new = G_s * one_hot_G_s

        # G_new = one_hot_G
        # G_s_new = one_hot_G_s
        # print("G_new")
        # print(G_new)
        G_new = torch.where(torch.sum(G_new, dim=0) == 0, 0, G_new / torch.sum(G_new, dim=0))
        G_s_new = torch.where(torch.sum(G_s_new, dim=0) == 0, 0, G_s_new / torch.sum(G_s_new, dim=0))
        # print("G_new")
        # print(G_new)

        count += 1
        prev_mu = mu
        prev_C = C
        prev_mu_s = mu_s
        prev_C_s = C_s
        mu = torch.tensor(compute_mu(xs, classes, Dist, reg, G_new))
        C = distance_to_centroid(xs, mu, Dist, reg)
        mu_s = torch.tensor(compute_mu(xs, classes, Dist, reg, G_s_new))
        C_s = distance_to_centroid(xs, mu_s, Dist, reg)

        # G_list.append(np.array(G))
        # mu_list.append(np.array(mu))

    return G, G_s, prev_C, prev_C_s, prev_mu, prev_mu_s


def main():
    # torch.set_printoptions(threshold=np.inf)
    # numList = [3, 6, 9, 12, 15, 15, 12, 9, 6, 3]
    numList = np.ones((10,), dtype=int) * 12
    train_data_fig, train_data_label, test_data_fig, test_data_label = load(numList)
    print("train_data_label", train_data_label.shape)
    print("train_data_fig", train_data_fig.shape)
    # print("origin train_fig[0]: ", np.sum(train_data_fig[0]))
    # plt.imshow(train_data_fig[0].reshape((28, 28)))
    # plt.show()
    train_data_fig = (train_data_fig + 1e-10) / np.sum(train_data_fig, axis=1).reshape(-1, 1)
    print("train_fig: ", train_data_fig.shape)
    print("train_fig[0]: ", np.sum(train_data_fig[0]))

    # Dist为代价矩阵
    Dist = ot.dist(position, position)
    print("Dist: ", Dist.shape)

    reg = 1e-3
    numClass = 16
    numIter = 4
    torch.random.manual_seed(20230816)
    np.random.seed(20230816)
    # fig_index = np.random.permutation(range(train_data_fig.shape[0]))
    # train_data_fig = train_data_fig[fig_index]
    # train_data_label = train_data_label[fig_index]

    lab = np.random.randint(0, train_data_fig.shape[0], numClass)
    print("lab:")
    print(lab)
    print(train_data_label[lab])
    # mu0 = train_data_fig[lab]
    t0 = time.time()
    mu0 = initial_center(train_data_fig, numClass, Dist, reg)
    C0 = distance_to_centroid(train_data_fig, mu0, Dist, reg)
    t1 = time.time()
    print("initial: ", t1 - t0)
    # print("***********{mu0}***********")
    # print(mu0)
    # print((mu0 == mu0[0]).all())
    print("*************C*************")
    print(C0)

    t2 = time.time()
    G, Gs, C, Cs, mu, mus = calculate_func(C0, torch.tensor(mu0, dtype=torch.float32),
                                           torch.tensor(train_data_fig, dtype=torch.float32), numClass,
                                           torch.tensor(Dist, dtype=torch.float32), reg,
                                           numIter=numIter, func_type='sinkhorn')
    t3 = time.time()
    print("time: ", t3 - t2)

    cen, y, interia = k_means(train_data_fig, numClass)
    t4 = time.time()
    print("kmeans time: ", t4 - t3)
    print(y)
    print(train_data_label)
    idx = np.argmax(G, axis=1)
    print(idx)

    # for i in range(6):
    #     # print(G_list[i])
    #     print(f"sum of col of G{i}:")
    #     print(np.sum(G_list[i], axis=0))

    np.save("./result/MNIST/Gw_1e-3_gauss_inv", G)
    np.save("./result/MNIST/muw_1e-3_gauss_inv", mu)
    # np.save("./result/MNIST/C_w_ablation", C)
    # np.save("./result/MNIST/Cs_w_ablation", Cs)
    # np.save("./result/MNIST/mu_w_ablation", mu)
    # np.save("./result/MNIST/mus_w_ablation", mus)
    # np.save("./result/MNIST/kmeans_mu_100", cen)
    # np.save("./result/MNIST/kmeans_y_100", y)


if __name__ == '__main__':
    main()
