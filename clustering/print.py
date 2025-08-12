import matplotlib.pyplot as plt
import os
import numpy as np
from mnist_loader import load
from sklearn.cluster import k_means

data_path = "./result/MNIST"
Gw = np.load(os.path.join(data_path, "Gw_1e-3_list.npy"))
Ge = np.load(os.path.join(data_path, "G_e_100.npy"))
Cw = np.load(os.path.join(data_path, "C_w_100.npy"))
Ce = np.load(os.path.join(data_path, "C_e_100.npy"))
muw = np.load(os.path.join(data_path, "muw_1e-3_list.npy"))
mue = np.load(os.path.join(data_path, "mu_e_100.npy"))
cen = np.load(os.path.join(data_path, "kmeans_mu_100.npy"))
y_label = np.load(os.path.join(data_path, "kmeans_y_100.npy"))

# numList = [3, 6, 9, 12, 15, 15, 12, 9, 6, 3]
numList = np.ones((10,), dtype=int) * 12
train_data_fig, train_data_label, test_data_fig, test_data_label = load(numList)
train_data_fig = (train_data_fig + 1e-10) / np.sum(train_data_fig, axis=1).reshape(-1, 1)

numTau = 6
numClass = 16
x = range(numClass)
tau_list = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
plt.figure(figsize=(24, 5))
for k in range(numTau):
    idx = np.argmax(Gw[k], axis=1)
    resultw = []
    resultw_label = []
    num_of_w_class = []
    for i in range(numClass):
        print(f"num of class {i}:")
        print(len(np.where(idx == i)[0]))
        num_of_w_class.append(len(np.where(idx == i)[0]))
        resultw.append(np.where(idx == i)[0])
        resultw_label.append(train_data_label[np.where(idx == i)[0]])

    resultw = np.array(resultw)
    resultw_label = np.array(resultw_label)
    print("resultw.shape: ", resultw.shape)
    print("resultw_label.shape: ", resultw_label.shape)

    totalw = np.zeros((numClass, numClass))
    for i in range(numClass):
        for j in range(numClass):
            totalw[i, j] = len(np.where(resultw_label[i] == j)[0])

    print("total:")
    print(totalw)

    acc = np.sum(np.max(totalw, axis=1)) / np.sum(totalw)
    print("acc:")
    print(acc)

    fig_w = np.zeros((4*28, 4*28))
    for i in range(numClass):
        n = int(i / 4)
        m = i % 4
        fig_w[(n * 28):(n * 28 + 28), (m * 28):(m * 28 + 28)] = muw[k][i].reshape(28, 28)

    plt.subplot(2, 6, k+7)
    plt.imshow(fig_w)
    plt.axis('off')
    # plt.title(f'tau={tau_list[k]}')
    # plt.imshow(fig_w)

    plt.subplot(2, 6, 1+k)
    plt.bar(x, height=num_of_w_class, width=0.4, color='green', label="ours(Wasserstein)")
    plt.plot(x, np.full_like(x, (1-tau_list[k])*120/16, dtype=float), color='black', linestyle='--')
    plt.plot(x, np.full_like(x, (1+tau_list[k])*120/16, dtype=float), color='black', linestyle='--', label='bounds')
    plt.title(f'tau={tau_list[k]}')
    plt.yticks(fontsize=8)

plt.show()

################################################################
idx = np.argmax(Ge, axis=1)
resulte = []
resulte_label = []
num_of_e_class = []
for i in range(numClass):
    print(f"num of class {i}:")
    print(len(np.where(idx == i)[0]))
    num_of_e_class.append(len(np.where(idx == i)[0]))
    resulte.append(np.where(idx == i)[0])
    resulte_label.append(train_data_label[np.where(idx == i)[0]])

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

for i in range(numClass):
    plt.figure(i)
    plt.imshow(mue[i].reshape((28, 28)))
    # plt.title(f"mu[{i}]")
    plt.axis('off')

############################################################
num_of_kmeans_class = []
result_kmeans = []
result_kmeans_label = []
for i in range(numClass):
    print(f"num of class {i}:")
    print(len(np.where(y_label == i)[0]))
    num_of_kmeans_class.append(len(np.where(y_label == i)[0]))
    result_kmeans.append(np.where(y_label == i)[0])
    result_kmeans_label.append(train_data_label[np.where(y_label == i)[0]])

total = np.zeros((numClass, numClass))
for i in range(numClass):
    for j in range(numClass):
        total[i, j] = len(np.where(result_kmeans_label[i] == j)[0])

print("total:")
print(total)

acc = np.sum(np.max(total, axis=1)) / np.sum(total)
print("acc:")
print(acc)

'''
plt.figure(figsize=(15, 15))
plt.subplot(5, 20, 1)
plt.imshow(mu[0].reshape(28, 28))
plt.axis('off')
for i in range(len(result[0])):
    plt.subplot(5, 20, 2 + i)
    plt.imshow(train_data_fig[result[0][i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 21)
plt.imshow(mu[1].reshape(28, 28))
plt.axis('off')
for i in range(len(result[1])):
    plt.subplot(5, 20, 22 + i)
    plt.imshow(train_data_fig[result[1][i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 41)
plt.imshow(mu[2].reshape(28, 28))
plt.axis('off')
for i in range(len(result[2])):
    plt.subplot(5, 20, 42 + i)
    plt.imshow(train_data_fig[result[2][i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 61)
plt.imshow(mu[3].reshape(28, 28))
plt.axis('off')
for i in range(len(result[3])):
    plt.subplot(5, 20, 62 + i)
    plt.imshow(train_data_fig[result[3][i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 81)
plt.imshow(mu[4].reshape(28, 28))
plt.axis('off')
for i in range(len(result[4])):
    plt.subplot(5, 20, 82 + i)
    plt.imshow(train_data_fig[result[4][i]].reshape(28, 28))
    plt.axis('off')

################################################################################

plt.figure(figsize=(20, 10))
plt.subplot(5, 20, 20)
plt.imshow(cen[0].reshape(28, 28))
plt.axis('off')
for i in range(len(np.where(y_label == 0)[0])):
    plt.subplot(5, 20, 19 - i)
    plt.imshow(train_data_fig[np.argwhere(y_label == 0)[i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 40)
plt.imshow(cen[1].reshape(28, 28))
plt.axis('off')
for i in range(len(np.where(y_label == 1)[0])):
    plt.subplot(5, 20, 39 - i)
    plt.imshow(train_data_fig[np.argwhere(y_label == 1)[i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 60)
plt.imshow(cen[2].reshape(28, 28))
plt.axis('off')
for i in range(len(np.where(y_label == 2)[0])):
    plt.subplot(5, 20, 59 - i)
    plt.imshow(train_data_fig[np.argwhere(y_label == 2)[i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 80)
plt.imshow(cen[3].reshape(28, 28))
plt.axis('off')
for i in range(len(np.where(y_label == 3)[0])):
    plt.subplot(5, 20, 79 - i)
    plt.imshow(train_data_fig[np.argwhere(y_label == 3)[i]].reshape(28, 28))
    plt.axis('off')
plt.subplot(5, 20, 100)
plt.imshow(cen[4].reshape(28, 28))
plt.axis('off')
for i in range(len(np.where(y_label == 4)[0])):
    plt.subplot(5, 20, 99 - i)
    plt.imshow(train_data_fig[np.argwhere(y_label == 4)[i]].reshape(28, 28))
    plt.axis('off')
'''

# x = [0, 1, 2, 3, 4]
# kmeans_id = sorted(range(len(num_of_kmeans_class)), key=lambda k: num_of_kmeans_class[k], reverse=False)
# e_id = sorted(range(len(num_of_e_class)), key=lambda k: num_of_e_class[k], reverse=False)
# w_id = sorted(range(len(num_of_w_class)), key=lambda k: num_of_w_class[k], reverse=False)
# print("num of w class", num_of_w_class)
# print("w id", w_id)

# num_of_kmeans_class.sort()
# num_of_e_class.sort()
# num_of_w_class.sort()

kmeans_id = [5, 3, 7, 9, 11, 0, 4, 6, 10, 14, 1, 13, 12, 2, 15, 8]
e_id = [11, 9, 1, 3, 7, 2, 15, 12, 5, 6, 0, 13, 4, 10, 14, 8]
w_id = [4, 10, 6, 12, 13, 3, 11, 0, 15, 7, 8, 2, 14, 1, 5, 9]

num_of_kmeans_class = np.array(num_of_kmeans_class)
num_of_kmeans_class = num_of_kmeans_class[kmeans_id]
num_of_e_class = np.array(num_of_e_class)
num_of_e_class = num_of_e_class[e_id]
num_of_w_class = np.array(num_of_w_class)
num_of_w_class = num_of_w_class[w_id]

plt.figure(figsize=(20, 5))
# plt.hist(total, bins=25, density=True, histtype='bar')
plt.subplot(3, 4, 1)
plt.bar(x, height=num_of_kmeans_class, width=0.3, alpha=0.8, color='red', label="kmeans")
plt.yticks(fontsize=14)
plt.legend()
plt.subplot(3, 4, 5)
plt.bar(x, height=num_of_e_class, width=0.3, color='green', label="ours(Euclidean)")
plt.plot(x, np.full_like(x, 6), color='black', linestyle='--')
plt.plot(x, np.full_like(x, 9), color='black', linestyle='--', label='bounds')
# plt.ylim([4, 13])
plt.yticks(fontsize=14)
plt.legend(loc=3)
plt.subplot(3, 4, 9)
plt.bar(x, height=num_of_w_class, width=0.3, color='blue', label="ours(Wasserstein)")
plt.plot(x, np.full_like(x, 6), color='black', linestyle='--')
plt.plot(x, np.full_like(x, 9), color='black', linestyle='--', label='bounds')
# plt.legend(loc=6, bbox_to_anchor=(-0.7, 0.5))
plt.yticks(fontsize=14)
plt.legend()

fig_kmeans = np.zeros((4*28, 4*28))
fig_e = np.zeros((4*28, 4*28))
fig_w = np.zeros((4*28, 4*28))

for i in range(numClass):
    n = int(i / 4)
    m = i % 4
    fig_kmeans[(n*28):(n*28+28), (m*28):(m*28+28)] = cen[kmeans_id[i]].reshape(28, 28)
    fig_e[(n * 28):(n * 28 + 28), (m * 28):(m * 28 + 28)] = mue[e_id[i]].reshape(28, 28)
    fig_w[(n * 28):(n * 28 + 28), (m * 28):(m * 28 + 28)] = muw[w_id[i]].reshape(28, 28)

plt.subplot(1, 4, 2)
plt.imshow(fig_kmeans)
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(fig_e)
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(fig_w)
plt.axis('off')

print(np.sum(Gw, axis=0))
print(np.sum(Gw, axis=0) * 120)

plt.show()
