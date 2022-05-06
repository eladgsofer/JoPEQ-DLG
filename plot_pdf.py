import matplotlib as plt
import pickle
import numpy as np

def plot_graphs(algo, iteration_list):
    with open('output/ITER_MAT_LOSS_' + algo + '_VANILLA.npy', 'rb') as f:
        dlg_loss_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_MSE_' + algo + '_VANILLA.npy', 'rb') as f:
        dlg_mse_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_SSIM_' + algo + '_VANILLA.npy', 'rb') as f:
        dlg_ssim_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_LOSS_' + algo + '_JOPEQ.npy', 'rb') as f:
        jopeq_loss_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_MSE_' + algo + '_JOPEQ.npy', 'rb') as f:
        jopeq_mse_per_iter_matrix = pickle.load(f)
    with open('output/ITER_MAT_SSIM_' + algo + '_JOPEQ.npy', 'rb') as f:
        jopeq_ssim_per_iter_matrix = pickle.load(f)
    with open('output/ITER_GRAD_MAT_NORM_' + algo + '_new.npy', 'rb') as f:
        grads_norm_mat = pickle.load(f)

    font = {'weight': 'bold', 'size': 12}
    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_loss_per_iter_matrix), axis=1),'-o',linewidth=1.5)
    plt.plot(iteration_list, np.mean(np.log(jopeq_loss_per_iter_matrix), axis=1), '-*', linewidth=1.5, markersize=8)
    plt.legend(['Vanilla DLG', 'JoPEQ'])
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("log(loss)")
    plt.savefig("log_loss_dlg_vs_jopeq.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_mse_per_iter_matrix), axis=1), linewidth=3)
    plt.plot(iteration_list, np.mean(np.log(jopeq_mse_per_iter_matrix), axis=1), linewidth=3)
    plt.title("dlg vanilla MSE vs JoPEQ MSE")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("log(MSE)")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(dlg_ssim_per_iter_matrix, axis=1), '-o', linewidth=1.5)
    plt.plot(iteration_list, np.mean(jopeq_ssim_per_iter_matrix, axis=1), '-*', linewidth=1.5, markersize=8)
    plt.legend(['Vanilla DLG', 'JoPEQ'])
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("SSIM")
    plt.savefig("ssim_dlg_vs_jopeq.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3)

    plt.show()