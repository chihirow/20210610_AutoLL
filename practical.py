import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import plot_results as pr
import torch
from torch import optim
import net
import common as cmn
import networkx as nx
import urllib.request as urllib
import io
import zipfile


def main(f_data, n_gpu):
    start = time.time()
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.random.seed(0)
    if n_gpu >= 0:
        torch.manual_seed(0)

    ################################################################################################
    if f_data == 1:
        f_directed = 0
        n_epoch = 10000
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
        sock = urllib.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()
        zf = zipfile.ZipFile(s)  # zipfile object
        gml = zf.read("football.gml").decode()  # read gml data
        gml = gml.split("\n")[1:]
        G = nx.parse_gml(gml)  # parse gml data
        A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
        n_batch0 = 5000
        f_us = False
    elif f_data == 2:  # Neural network, http://www-personal.umich.edu/~mejn/netdata/
        f_directed = 1
        n_epoch = 5000
        G = nx.read_gml('celegansneural/celegansneural.gml')
        A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
        n_batch0 = 3000
        f_us = True
    fig_size = np.array([15, 15])
    # ---------------------------------------------------------------------------------------------
    lr = 1e-2
    lambda_reg = 1e-10
    A = (A - np.min(A)) / (np.max(A) - np.min(A))  # Make all the entries 0 <= A_ij <= 1
    n = A.shape[0]
    # ---------------------------------------------------------------------------------------------
    if f_directed == 0:  # undirected
        n_units_in = np.array([n, 10, 1])  # 100
        n_units_out = np.array([2, 10, 1])
    else:  # directed
        n_units_in = np.array([2 * n, 10, 1])  # 100
        n_units_out = np.array([2, 10, 1])
    if f_us:
        r_zero = 8  # Apply under-sampling of zeros to make #0:#1 = r_zero:1
    clr_matrix = 'CMRmap_r'
    ################################################################################################

    n_batch = np.min([n_batch0, n ** 2])  # Batch size
    n_iter = np.int64(np.ceil(np.float64(n_epoch * n * n) / n_batch))  # No. of epochs
    print('n_iter: ' + str(n_iter))
    print('Matrix size: ' + str(n) + ' x ' + str(n))

    # Apply under-sampling of zeros
    if f_us:
        n_0_original = np.sum(A == 0)
        n_1 = n ** 2 - n_0_original  # No. of non-zero entries
        n_0 = np.ceil(n_1 * r_zero).astype(np.int64)
        A_vector = np.reshape(A, n ** 2)
        idx_0_original = np.where(A_vector == 0)[0]
        idx_0_selected = np.random.choice(n_0_original, n_0, replace=False)
        idx_0 = idx_0_original[idx_0_selected]  # Choose n_zero entries
        idx_1 = np.where(A_vector != 0)[0]
        idx_us = np.append(idx_0, idx_1)
        print('n_0_original = ' + str(n_0_original) + ', n_1 = ' + str(n_1) + ', n_0 = ' + str(n_0))
        print('n_0 + n_1 = ' + str(n_0 + n_1) + ', n_batch = ' + str(n_batch))

    pr.plot_A_practical(A, fig_size, clr_matrix, 'practical' + str(f_data) + '_input', r'Observed matrix $A$')

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
        ##############################################################################
        if f_us:
            idx_entry = np.random.choice(idx_us, n_batch, replace=False)
        else:
            idx_entry = np.random.choice(n ** 2, n_batch, replace=False)
        ##############################################################################
        x_row, x_col, y = cmn.define_xy(A, idx_entry, n_batch, device, f_directed)
        loss_t = loss.calc_loss(x_row, x_col, y, lambda_reg)
        model.zero_grad()
        loss_t.backward()
        optimizer.step()
        loss_all[t] = loss_t.data
        if t / n_iter >= cnt / 30:
            pr.plot_loss(loss_all, 'practical' + str(f_data))  # Plot training loss
            print('>', end='', flush=True)
            cnt += 1
    print('')
    pr.plot_loss(loss_all, 'practical' + str(f_data))  # Plot training loss

    # Plot results
    model.eval()
    A_out, h, order = cmn.calc_result(A, model, device, f_directed)
    pr.plot_latent(h, 'Node features', 'practical' + str(f_data) + '_features')
    pr.plot_latent(h[order], 'Reordered node features', 'practical' + str(f_data) + '_features_sort')
    pr.plot_A_practical(A[order, :][:, order], fig_size, clr_matrix,
                        'practical' + str(f_data) + '_input_sort', 'Reordered input matrix\n(proposed AutoLL)')
    pr.plot_A_practical(A_out[order, :][:, order], fig_size, clr_matrix,
                        'practical' + str(f_data) + '_out', 'Reordered output matrix\n(proposed AutoLL)')
    elapsed_time = time.time() - start
    print('Overall computation time :{:.2f}'.format(elapsed_time) + '[sec]')

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main(int(sys.argv[1]), int(sys.argv[2]))
