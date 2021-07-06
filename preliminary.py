import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import math, sys, time
import gen_matrix as gm
import plot_results as pr
import torch
from torch import optim
import net
import common as cmn


def main(f_structure, f_directed, n_gpu):
    # f_structure = 1: LBM, 2: Oblique gradation model
    # f_directed: 0: undirected, 1: directed
    start = time.time()
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.random.seed(0)
    if n_gpu >= 0:
        torch.manual_seed(0)

    ################################################################################################
    n = 120
    if f_structure == 1:  # Latent Block Model
        B = np.array([
            [0.9, 0.1, 0.3],
            [0.4, 0.8, 0.2],
            [0.1, 0.3, 0.7]])
        S = 0.05 * np.ones((B.shape[0], B.shape[1]))
        A = gm.gen_matrix(n, B, S, 1, f_directed, 0)  # Gaussian case
    elif f_structure == 2:  # Oblique gradation model
        B = np.array([0.9, 0.1])
        S = 0.05
        A, _ = gm.gen_matrix2(n, B, S, 1, f_directed, 0)
    n_epoch = 200  # 1000
    lr = 1e-2  # Learning rate
    lambda_reg = 1e-10  # Ridge regularization hyperparameter
    n_batch0 = 200  # 2000
    # -------------------------------------------------------------------------------------------
    A = (A - np.min(A)) / (np.max(A) - np.min(A))  # Make all the entries 0 <= A_ij <= 1
    # ---------------------------------------------------------------------------------------------
    if f_directed == 0:  # undirected
        n_units_in = np.array([n, 10, 1])
        n_units_out = np.array([2, 10, 1])
        str_directed = 'u'
    else:  # directed
        n_units_in = np.array([2 * n, 10, 1])
        n_units_out = np.array([2, 10, 1])
        str_directed = 'd'
    clr_matrix = 'CMRmap_r'
    ################################################################################################

    n_batch = np.min([n_batch0, n ** 2])  # Batch size
    n_iter = np.int64(np.ceil(np.float64(n_epoch * n * n) / n_batch))  # No. of epochs
    print('n_iter: ' + str(n_iter))
    print('Matrix size: ' + str(n) + ' x ' + str(n))

    A_bar = np.copy(A)
    pr.plot_A(A_bar, r'Matrix $\bar{A}$', 'synthetic' + str(f_structure) + str_directed + '_input', clr_matrix)
    g = np.random.permutation(n)
    A = A[g, :][:, g]  # Random permutation of rows and columns
    pr.plot_A(A, r'Observed matrix $A$', 'synthetic' + str(f_structure) + str_directed + '_input_permutated',
              clr_matrix)

    # Define NN model
    model = net.AutoLL(n_units_in, n_units_out)
    print('No. of params: {}'.format(sum(prm.numel() for prm in model.parameters())) +
          ' (total), {}'.format(sum(prm.numel() for prm in model.parameters() if prm.requires_grad)) + ' (learnable)')
    loss = net.Loss(model)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    device = torch.device('cuda:{}'.format(n_gpu)) if n_gpu >= 0 else torch.device('cpu')
    torch.cuda.set_device(device)
    model.to(device)
    model.train()

    # NN Training
    loss_all = np.full(n_iter, np.nan)
    cnt = 1
    for t in range(n_iter):
        idx_entry = np.random.choice(n ** 2, n_batch, replace=False)
        x_row, x_col, y = cmn.define_xy(A, idx_entry, n_batch, device, f_directed)
        loss_t = loss.calc_loss(x_row, x_col, y, lambda_reg)
        model.zero_grad()
        loss_t.backward()
        optimizer.step()
        loss_all[t] = loss_t.data
        if t / n_iter >= cnt / 30:
            pr.plot_loss(loss_all, 'synthetic' + str(f_structure) + str_directed)  # Plot training loss
            print('>', end='', flush=True)
            cnt += 1
    print('')
    pr.plot_loss(loss_all, 'synthetic' + str(f_structure) + str_directed)  # Plot training loss

    # Plot results
    model.eval()
    A_out, h, order = cmn.calc_result(A, model, device, f_directed)
    pr.plot_latent(h, 'Node features', 'synthetic' + str(f_structure) + str_directed + '_features')
    pr.plot_latent(h[order],
                   'Reordered node features', 'synthetic' + str(f_structure) + str_directed + '_features_sort')

    pr.plot_A(A[order, :][:, order], 'Reordered input matrix\n(proposed AutoLL)',
              'synthetic' + str(f_structure) + str_directed + '_input_sort', clr_matrix)
    pr.plot_A(A_out[order, :][:, order], 'Reordered output matrix\n(proposed AutoLL)',
              'synthetic' + str(f_structure) + str_directed + '_out', clr_matrix)
    elapsed_time = time.time() - start
    print('Overall computation time :{:.2f}'.format(elapsed_time) + '[sec]')

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
