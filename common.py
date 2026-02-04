import numpy as np
import math
import torch


def define_xy(A, idx_entry, n_batch, device, f_directed):
    n = A.shape[0]
    idx_r = idx_entry // n
    idx_c = idx_entry % n
    if f_directed == 0:  # undirected
        x_row_cpu = np.zeros((n_batch, n))
        x_col_cpu = np.zeros((n_batch, n))
        y = np.zeros((n_batch, 1))
        for i in range(n_batch):
            x_row_cpu[i, :] = np.copy(A[idx_r[i], :])  # Row information
            x_col_cpu[i, :] = np.copy(A[:, idx_c[i]])  # Column information
            y[i, :] = np.copy(A[idx_r[i], idx_c[i]])
    else:  # directed
        x_row_cpu = np.zeros((n_batch, 2 * n))
        x_col_cpu = np.zeros((n_batch, 2 * n))
        y = np.zeros((n_batch, 1))
        for i in range(n_batch):
            x_row_cpu[i, :] = np.append(np.copy(A[idx_r[i], :]), np.copy(A[:, idx_r[i]]))  # Row & column information
            x_col_cpu[i, :] = np.append(np.copy(A[idx_c[i], :]), np.copy(A[:, idx_c[i]]))  # Row & column information
            y[i, :] = np.copy(A[idx_r[i], idx_c[i]])
    x_row = torch.tensor(x_row_cpu).to(device, dtype=torch.float)
    x_col = torch.tensor(x_col_cpu).to(device, dtype=torch.float)
    y = torch.tensor(y).to(device, dtype=torch.float)

    return x_row, x_col, y


def calc_result(A, model, device, f_directed):
    n = A.shape[0]
    A_out = np.zeros((n, n))
    h = np.zeros(n)  # Latent feature
    for i in range(n):
        cnt_i = 0
        for j in range(n):
            if f_directed == 0:  # undirected
                if i <= j:
                    x_row_cpu = np.copy(A[i, :])
                    x_row_cpu = x_row_cpu[np.newaxis, :]
                    x_col_cpu = np.copy(A[:, j])
                    x_col_cpu = x_col_cpu[np.newaxis, :]
                    x_row = torch.tensor(x_row_cpu).to(device, dtype=torch.float)
                    x_col = torch.tensor(x_col_cpu).to(device, dtype=torch.float)
                    with torch.no_grad():
                        A_out_ij, h_i, _ = model(x_row, x_col)
                        A_out[i, j] = A_out_ij.detach().cpu().clone().numpy()
                        A_out[j, i] = A_out[i, j]
                        if cnt_i == 0:
                            h[i] = h_i.detach().cpu().clone().numpy()
                            cnt_i += 1
            else:  # directed
                x_row_cpu = np.append(np.copy(A[i, :]), np.copy(A[:, i]))
                x_row_cpu = x_row_cpu[np.newaxis, :]
                x_col_cpu = np.append(np.copy(A[j, :]), np.copy(A[:, j]))
                x_col_cpu = x_col_cpu[np.newaxis, :]
                x_row = torch.tensor(x_row_cpu).to(device, dtype=torch.float)
                x_col = torch.tensor(x_col_cpu).to(device, dtype=torch.float)
                with torch.no_grad():
                    A_out_ij, h_i, _ = model(x_row, x_col)
                    A_out[i, j] = A_out_ij.detach().cpu().clone().numpy()
                    if cnt_i == 0:
                        h[i] = h_i.detach().cpu().clone().numpy()
                        cnt_i += 1
    order = np.argsort(h)

    return A_out, h, order


def mr_pca_row(A):  # [Friendly2002]
    n = A.shape[0]
    p = A.shape[1]
    A_copy = np.copy(A)
    for i in range(n):
        A_copy[i, :] = (A_copy[i, :] - np.mean(A_copy[i, :])) / np.std(A_copy[i, :] + 1e-5)
    R = (1 / p) * np.matmul(A_copy, A_copy.T)
    _, _, U = np.linalg.svd(R, full_matrices=True)
    # u1 = U[:, 0]
    # u2 = U[:, 1]
    u1 = U[0, :]  # 2026/2/4
    u2 = U[1, :]  # 2026/2/4
    alpha = np.zeros(n)  # -pi/2 <= <= 3/2 pi
    for i in range(n):
        if u1[i] <= 0:
            alpha[i] = np.arctan(u2[i] / (u1[i] + 1e-5)) + math.pi
        else:
            alpha[i] = np.arctan(u2[i] / (u1[i] + 1e-5))
    idx = np.argsort(alpha)
    alpha_sort = alpha[idx]
    d_alpha = np.zeros(n)
    for i in range(n):
        if i == 0:
            d_alpha[i] = 2 * math.pi + alpha_sort[i] - alpha_sort[i - 1]
        else:
            d_alpha[i] = alpha_sort[i] - alpha_sort[i - 1]
    idx_max = np.argmax(d_alpha)
    idx[idx < idx_max] += n
    idx -= idx_max

    return idx  # row order, A[idx, :]


def mr_svd(A):
    U, D, V = np.linalg.svd(A, full_matrices=True)
    # u1 = U[0, :]
    # v1 = V[:, 0]
    u1 = U[:, 0]  # 2026/2/4
    v1 = V[0, :]  # 2026/2/4
    idx_row = np.argsort(u1)
    idx_col = np.argsort(v1)
    A_approx = D[0] * np.matmul(u1[:, np.newaxis], v1[np.newaxis, :])  # rank-one approximation of A

    return idx_row, idx_col, A_approx


def mr_mds_row(A):
    n = A.shape[0]
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D2[i, j] = np.sum((A[i, :] - A[j, :]) ** 2)
    J = np.eye(n) - (1 / n) * np.ones((n, n))
    B = - (1 / 2) * np.matmul(J, np.matmul(D2, J))
    w, V = np.linalg.eig(B)
    idx = np.argsort(-w.real)
    v1 = V[:, idx[0]].real
    idx_row = np.argsort(v1)

    return idx_row


def select_order(order, P, P_bar):
    n = P.shape[0]
    order_0 = np.copy(order)
    order_list = np.append(order_0[np.newaxis, :], np.flip(order_0)[np.newaxis, :], axis=0)
    err0 = np.inf
    for i in range(2):
        err = np.sum((P_bar - P[order_list[i], :][:, order_list[i]]) ** 2)
        if err < err0:
            i_order = i
            err0 = err
    if i_order == 0:
        order_opt = order_0
    else:
        order_opt = np.flip(order_0)

    err = np.sum((P_bar - P[order_opt, :][:, order_opt]) ** 2) / (n ** 2)

    return order_opt, err
