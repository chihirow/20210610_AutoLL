import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.special as spys
import math


def plot_loss(loss_all, str_file):
    plt.rcParams['font.size'] = 25
    plt.figure(figsize=(12, 4))
    x = np.arange(0, loss_all.shape[0], 1)
    plt.plot(x, loss_all, color=np.zeros(3), label='Overall training loss')
    plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
    plt.tick_params(bottom=True, left=True, right=False, top=False)
    plt.xlabel('Iterations')
    plt.ylabel('Training loss')
    plt.savefig('result/loss_' + str_file + '.png', bbox_inches='tight')
    plt.close()


def plot_latent(h, str_title, str_file):
    ylim_margin = 0.1
    n = h.shape[0]
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, n + 1, 1), h, color='k', ls='None', marker='.')
    plt.xlabel('Nodes')
    plt.title(str_title)
    plt.ylim([-ylim_margin, 1 + ylim_margin])
    plt.tight_layout()
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_A(A, str_title, str_file, clr_matrix):
    n = A.shape[0]
    p = A.shape[1]
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(6, 5))
    plt.imshow(A, cmap=clr_matrix, vmin=0, vmax=1, interpolation='none')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.colorbar(ticks=np.arange(0, 1.1, 0.2))
    plt.xlim([-0.5, p - 0.5])
    plt.ylim([-0.5, n - 0.5])
    plt.gca().set_aspect('equal')
    plt.title(str_title)
    plt.gca().invert_yaxis()
    plt.savefig('result/A_' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/A_' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_A_ylabel(color_y1, color_y0, str_class, str_file):
    plt.rcParams["font.size"] = 25
    plt.figure(figsize=(6, 12))
    p_height = 0.2
    p_width = 0.2
    plt.text(0, p_height, r'$\bullet$', color=color_y1, verticalalignment='center', fontsize=50)
    plt.text(p_width, p_height, str_class[1], verticalalignment='center')
    plt.text(0, -p_height, r'$\bullet$', color=color_y0, verticalalignment='center', fontsize=50)
    plt.text(p_width, -p_height, str_class[0], verticalalignment='center')
    plt.xlim([-0.2, 1])
    plt.ylim(15 * p_height * np.array([-1, 1]))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.axis('off')
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_A_practical(A, fig_size, clr_matrix, str_file, str_title):
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(fig_size[0], fig_size[1]))
    plt.imshow(A, cmap=clr_matrix, aspect='auto', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(ticks=np.arange(0, 1.1, 0.2))
    plt.title(str_title)
    plt.xlim([-0.5, A.shape[1] - 0.5])
    plt.ylim([-0.5, A.shape[0] - 0.5])
    plt.gca().set_aspect('equal')

    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.gca().invert_yaxis()
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, f_ylog, str_directed):
    x_list = np.arange(1, S_list.shape[0] + 1, 1)
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_list, np.mean(err, axis=1), yerr=np.std(err, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(0 / 4), label='AutoLL')
    plt.errorbar(x_list, np.mean(err_svd, axis=1), yerr=np.std(err_svd, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(1 / 4), label='SVD-Rank-One')
    plt.errorbar(x_list, np.mean(err_pca, axis=1), yerr=np.std(err_pca, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(2 / 4), label='SVD-Angle')
    plt.errorbar(x_list, np.mean(err_mds, axis=1), yerr=np.std(err_mds, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(3 / 4), label='MDS')
    if f_ylog:
        plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xlabel(r'$t$')
    plt.title('Graph reordering error')
    plt.tight_layout()
    if f_ylog:
        plt.savefig('result/compare_mean_ylog_' + str_directed + '.png', bbox_inches='tight')
        plt.savefig('result/compare_mean_ylog_' + str_directed + '.eps', bbox_inches='tight')
        plt.savefig('result/compare_mean_ylog_' + str_directed + '.pdf', bbox_inches='tight')  # 2026/2/6
    else:
        plt.savefig('result/compare_mean_' + str_directed + '.png', bbox_inches='tight')
        plt.savefig('result/compare_mean_' + str_directed + '.eps', bbox_inches='tight')
        plt.savefig('result/compare_mean_' + str_directed + '.pdf', bbox_inches='tight')  # 2026/2/6
    plt.close()


def plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, f_ylog, str_directed):
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(12, 8))
    err2 = np.copy(err)
    err2[err2 == 0] = np.inf
    err2_pca = np.copy(err_pca)
    err2_pca[err2_pca == 0] = np.inf
    err2_svd = np.copy(err_svd)
    err2_svd[err2_svd == 0] = np.inf
    err2_mds = np.copy(err_mds)
    err2_mds[err2_mds == 0] = np.inf
    ylim0 = np.min([np.min(err2), np.min(err2_pca), np.min(err2_svd), np.min(err2_mds)]) * 0.9
    ylim1 = np.max([np.max(err), np.max(err_pca), np.max(err_svd), np.max(err_mds)]) * 1.1
    n_S = err.shape[0]
    n_replicate = err.shape[1]
    x_scatter = np.zeros(n_S * n_replicate)
    err_scatter = np.zeros(n_S * n_replicate)
    err_scatter_pca = np.zeros(n_S * n_replicate)
    err_scatter_svd = np.zeros(n_S * n_replicate)
    err_scatter_mds = np.zeros(n_S * n_replicate)
    for k in range(n_S):
        for m in range(n_replicate):
            x_scatter[n_replicate * k + m] = k + 1
            err_scatter[n_replicate * k + m] = err[k, m]
            err_scatter_pca[n_replicate * k + m] = err_pca[k, m]
            err_scatter_svd[n_replicate * k + m] = err_svd[k, m]
            err_scatter_mds[n_replicate * k + m] = err_mds[k, m]

    plt.subplot(1, 4, 1)
    plt.scatter(x_scatter, err_scatter, marker='.', color='k')
    plt.xlabel(r'$t$')
    plt.title('Graph reordering\nerror (AutoLL)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 2)
    plt.scatter(x_scatter, err_scatter_svd, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Graph reordering\nerror (SVD-Rank-One)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 3)
    plt.scatter(x_scatter, err_scatter_pca, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Graph reordering\nerror (SVD-Angle)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 4)
    plt.scatter(x_scatter, err_scatter_mds, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Graph reordering\nerror (MDS)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')

    plt.tight_layout()
    if f_ylog:
        plt.savefig('result/compare_ylog_' + str_directed + '.png', bbox_inches='tight')
        plt.savefig('result/compare_ylog_' + str_directed + '.eps', bbox_inches='tight')
        plt.savefig('result/compare_ylog_' + str_directed + '.pdf', bbox_inches='tight')  # 2026/2/6
    else:
        plt.savefig('result/compare_' + str_directed + '.png', bbox_inches='tight')
        plt.savefig('result/compare_' + str_directed + '.eps', bbox_inches='tight')
        plt.savefig('result/compare_' + str_directed + '.pdf', bbox_inches='tight')  # 2026/2/6
    plt.close()
