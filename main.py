# -*- coding: utf-8 -*-
import argparse
import numpy as np

import iDLG
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
print(torch.__version__, torchvision.__version__)
import os
os.system('')
from utils import label_to_onehot, cross_entropy_for_onehot
import random
from torch.distributions.laplace import Laplace
from vision import LeNet, CNN, weights_init
import copy
from dlg import dlg_cls, add_uveqFed, run_dlg
import sys
from datetime import datetime

tomer_path = r"C:\Users\tomer\Documents\Final_project_git\federated_learning_uveqfed_dlg\Federated-Learning-Natalie"
elad_path = r"/Users/elad.sofer/src/Engineering Project/federated_learning_uveqfed_dlg/Federated-Learning-Natalie"
sys.path.append(elad_path)
sys.path.append(tomer_path)

from models import LENETLayer
from federated_utils import PQclass
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='pick between - CIFAR100, CIFAR10.')

# Federated learning arguments
parser.add_argument('--R', type=int, default=16,
                    choices=[1, 2, 4],
                    help="compression rate (number of bits)")
parser.add_argument('--epsilon', type=float, default=500,
                    choices=[1, 5, 10],
                    help="privacy budget (epsilon)")
parser.add_argument('--dyn_range', type=float, default=1,
                    help="quantizer dynamic range")
parser.add_argument('--quantization_type', type=str, default='SDQ',
                    choices=[None, 'Q', 'DQ', 'SDQ'],
                    help="whether to perform (Subtractive) (Dithered) Quantization")
parser.add_argument('--quantizer_type', type=str, default='mid-tread',
                    choices=['mid-riser', 'mid-tread'],
                    help="whether to choose mid-riser or mid-tread quantizer")

parser.add_argument('--privacy_noise', type=str, default='laplace',
                    choices=[None, 'laplace', 'PPN'],
                    help="add the signal privacy preserving noise of type laplace or PPN")

parser.add_argument('--device', type=str, default='cpu',
                    choices=['cuda:0', 'cuda:1', 'cpu'],
                    help="device to use (gpu or cpu)")

parser.add_argument('--attack', type=str, default='JOPEQ',
                    choices=['JOPEQ', 'noise_only', 'quantization'],
                    help="DLG/iDLG attack type ")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
print("Running on %s" % device)


img_index = args.index




import iDLG
def produce_image_pentas(image_number_list,iteration_list,epsilon_list,bit_rate_lst):
    plt.xscale("log")
    loss_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])
    grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)


    now = datetime.now()


    dt_string = now.strftime("%Y%m%d%H%M")
    parent_path = "output/image_penta_run-"+str(dt_string)

    os.mkdir(parent_path)
    # run all the tests:
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()
    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            orig_img = dlg.load_image(n)
            dlg.compute_gradients()

            ## DLG ONLY ##
            dlg.dlg()
            dlg_only_img = dlg.final_image

            for k, bit_rate in enumerate(bit_rate_lst):

                ## QUANTIZATON ONLY ##
                args.attack = 'quantization'
                dlg.apply_noise(100, bit_rate, args=args)
                dlg.dlg()
                quant_only_img = dlg.final_image

                for l, epsilon in enumerate(epsilon_list):

                    ## JOPEQ ##
                    args.attack = 'JOPEQ'
                    args.privacy_noise = 'PPN'
                    dlg.apply_noise(epsilon, bit_rate, args=args)
                    dlg.dlg()
                    JOPEQ_only_img = dlg.final_image

                    ## LAPLACE ONLY ##
                    args.attack = 'noise_only'
                    args.privacy_noise = 'laplace'
                    dlg.apply_noise(epsilon, bit_rate, args=args)
                    dlg.dlg()
                    laplace_only_img = dlg.final_image
                    dir_path = parent_path+"/{0}_{1}_{2}_{3}".format(iter,bit_rate,epsilon,n)
                    os.mkdir(dir_path)
                    orig_img.save(dir_path+"/orig.jpg")
                    dlg_only_img.save(dir_path + "/dlg_only.jpg")
                    quant_only_img.save(dir_path + "/quantization_only.jpg")
                    JOPEQ_only_img.save(dir_path + "/JOPEQ.jpg")
                    laplace_only_img.save(dir_path + "/laplace_noise.jpg")






def train_model(image_number_list,iteration_list, algo='DLG'):
    plt.xscale("log")
    acc_vec = np.zeros([len(iteration_list)])
    grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True, transform=transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()
    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            acc_vec[i] = dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            dlg.load_image(n)
            gradients = dlg.compute_gradients()
            grads_norm_mat[i, j] = sum([x.norm(p=2) ** 2 for x in gradients]) ** (0.5)
    with open('output/TRAINING_TEST_GRADS'+algo+'.npy', 'wb') as f:
        pickle.dump(grads_norm_mat, f)
    with open('output/TRAINING_TEST_ACC' + algo + '.npy', 'wb') as f:
        pickle.dump(acc_vec, f)
    # plot the accuracy
    plt.figure()
    font = {
        'weight': 'bold',
        'size': 16}
    plt.rc('font', **font)
    plt.plot(iteration_list, acc_vec, linewidth=3)
    plt.title("Lenet acc after training the model")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("accuracy[%]")
    plt.figure()
    plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3)
    plt.title("gradients L2 Norm after training the model")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("mean L2 Norm")
    plt.show()



def run_iteration_dlg_idlg_tests(image_number_list,iteration_list, algo='DLG'):

    plt.xscale("log")
    # Vanilla DLG
    dlg_loss_per_iter_matrix = np.zeros([len(iteration_list),len(image_number_list)])
    dlg_mse_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])
    dlg_ssim_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])

    # JoPEQ
    jopeq_loss_per_iter_matrix = np.zeros([len(iteration_list),len(image_number_list)])
    jopeq_mse_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])
    jopeq_ssim_per_iter_matrix = np.zeros([len(iteration_list), len(image_number_list)])

    grads_norm_mat = np.zeros([len(iteration_list), len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    dlg = dlg_cls(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        noise_func=add_uveqFed)
    dlg.config_model()

    for i, iter in enumerate(iteration_list):
        print("iteration number {0}".format(i))
        if i > 0:
            dlg.train_model(1)
        for j, n in enumerate(image_number_list):
            dlg.load_image(n)
            gradients = dlg.compute_gradients()
            grads_norm_mat[i,j] = sum([x.norm(p=2) ** 2 for x in gradients]) ** (0.5)

            # Run Vanilla DLG
            loss, mse, ssim = dlg.dlg()

            dlg_loss_per_iter_matrix[i, j] = loss
            dlg_mse_per_iter_matrix[i,j] = mse
            dlg_ssim_per_iter_matrix[i, j] = ssim

            # Run DLG with JoPEQ noised gradients
            args.attack = 'JOPEQ'
            args.privacy_noise = 'PPN'
            dlg.apply_noise(333, 8, args=args)

            loss, mse, ssim = dlg.dlg()
            jopeq_loss_per_iter_matrix[i, j] = loss
            jopeq_mse_per_iter_matrix[i,j] = mse
            jopeq_ssim_per_iter_matrix[i, j] = ssim

        #loss_per_iter_matrix[i, j] = i+j
        # print("iter:{0} average loss: {1} loss values:{2}".format(iter,np.mean(loss_per_epsilon_matrix[i]),loss_per_epsilon_matrix[i]))

    # save the loss into a matrix

    with open('output/ITER_MAT_LOSS_'+algo+'_VANILLA.npy', 'wb') as f:
        pickle.dump(dlg_loss_per_iter_matrix, f)
    with open('output/ITER_MAT_MSE_'+algo+'_VANILLA.npy', 'wb') as f:
        pickle.dump(dlg_mse_per_iter_matrix, f)
    with open('output/ITER_MAT_SSIM_'+algo+'_VANILLA.npy', 'wb') as f:
        pickle.dump(dlg_ssim_per_iter_matrix, f)

    with open('output/ITER_MAT_LOSS_'+algo+'_JOPEQ.npy', 'wb') as f:
        pickle.dump(jopeq_loss_per_iter_matrix, f)
    with open('output/ITER_MAT_MSE_'+algo+'_JOPEQ.npy', 'wb') as f:
        pickle.dump(jopeq_mse_per_iter_matrix, f)
    with open('output/ITER_MAT_SSIM_'+algo+'_JOPEQ.npy', 'wb') as f:
        pickle.dump(jopeq_ssim_per_iter_matrix, f)

    with open('output/ITER_GRAD_MAT_NORM_'+algo+'_new.npy', 'wb') as f:
        pickle.dump(grads_norm_mat, f)
    # plot the accuracy

    font = {
        'weight': 'bold',
        'size': 16}
    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list,np.mean(np.log(dlg_loss_per_iter_matrix),axis=1),linewidth=3)
    plt.plot(iteration_list, np.mean(np.log(jopeq_loss_per_iter_matrix), axis=1),linewidth=3)
    plt.title("dlg vanilla ssim vs jopeq loss")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("epoches")
    plt.ylabel("loss")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list,np.mean (np.log(dlg_mse_per_iter_matrix),axis=1),linewidth=3)
    plt.plot(iteration_list, np.mean(np.log(jopeq_mse_per_iter_matrix), axis=1),linewidth=3)
    plt.title("dlg vanilla MSE vs JoPEQ MSE")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("epoches")
    plt.ylabel("log(MSE)")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list,np.mean(np.log(dlg_ssim_per_iter_matrix),axis=1),linewidth=3)
    plt.plot(iteration_list, np.mean(np.log(jopeq_ssim_per_iter_matrix), axis=1),linewidth=3)
    plt.title("dlg vanilla ssim vs jopeq ssim")
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("epoches")
    plt.ylabel("ssim")

    plt.figure()
    plt.plot(iteration_list,np.mean(grads_norm_mat,axis=1),linewidth=3)

    plt.show()


def run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list,bit_rate_lst, algo='DLG'):
    """

    Args:
        image_number_list:
        epsilon_list:
        algo:

    Returns:

    """
    plt.xscale("log")
    loss_per_epsilon_matrix = np.zeros([len(bit_rate_lst), len(epsilon_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for k, bit_rate in enumerate(bit_rate_lst):
        for i, epsilon in enumerate(epsilon_list):
            for j,n in enumerate(image_number_list):

                # extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
                dlg = dlg_cls(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    args=args,
                    noise_func=add_uveqFed)
                loss_per_epsilon_matrix[k, i, j] = dlg(
                    img_index=n,
                    learning_epoches=0,
                    read_grads=-1,
                    epsilon=epsilon,
                    bit_rate=bit_rate)
                # loss_per_epsilon_matrix[k,i, j] = k+i+j
                print("#### image {0} epsilon {1} bitRate {2} loss {3}####".format(j, epsilon, bit_rate,loss_per_epsilon_matrix[k,i,j]))
            print("bit_rate: {0} epsilon:{1} average loss: {2} loss values:{3}".format(bit_rate, epsilon,np.mean(loss_per_epsilon_matrix[k][i]),loss_per_epsilon_matrix[k][i]))

    # # save the loss into a matrix

    #     np.save(f, loss_per_epsilon_matrix[0,:,:])
    # np.savetxt('output/epsilon_mat'+algo+'.txt', loss_per_epsilon_matrix[0,:,:], fmt='%1.4e')

    with open('output/TOTAL_MAT'+algo+'.npy', 'wb') as f:
        pickle.dump(loss_per_epsilon_matrix, f)

    # # plot the accuracy
    # plt.figure()
    # font = {'weight': 'bold','size': 16}
    #
    # plt.rc('font', **font)
    # plt.plot(epsilon_list,np.mean(loss_per_epsilon_matrix,axis=1),linewidth=3)
    # plt.title("{0} loss attack type {1} for various levels of noise levels".format(algo, args.attack))
    # plt.grid(visible=True,axis="y")
    # plt.grid(visible=True,which='minor')
    # plt.xlabel("2/epsilon")
    # plt.ylabel("loss")
import pickle
def run_dlg_idlg_tests(image_number_list,check_point_list,model_number, algo='DLG'):
    plt.xscale("log")
    loss_per_iter_matrix = np.zeros([len(check_point_list),len(image_number_list)])
    # opening datasets
    dataset = getattr(datasets, args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset("~/.torch", train=False, download=True,transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])),
        batch_size=LeNet.BATCH_SIZE, shuffle=True)

    # run all the tests:
    for i,iter in enumerate(check_point_list):
        for j,n in enumerate(image_number_list):
            extract_img = run_dlg if algo == 'DLG' else iDLG.run_idlg
            loss_per_iter_matrix[i, j] = extract_img(n,
                                                        train_loader=train_loader,
                                                        test_loader=test_loader,
                                                        learning_epoches=0,
                                                        epsilon=0,
                                                        noise_func=add_uveqFed,
                                                        read_grads=iter,
                                                        model_number=model_number)
        #loss_per_epsilon_matrix[i, j] = i+j
        print("iter:{0} average loss: {1} loss values:{2}".format(iter,np.mean(loss_per_iter_matrix[i]),loss_per_iter_matrix[i]))

    # # save the loss into a matrix
    # with open('../output/loss_mat'+algo+'.npy', 'wb') as f:
    #     np.save(f, loss_per_iter_matrix)
    # np.savetxt('../output/loss_mat'+algo+'.txt', loss_per_iter_matrix, fmt='%1.4e')

    # plot the accuracy
    plt.figure()
    font = {
        'weight': 'bold',
        'size': 16}

    plt.rc('font', **font)
    plt.plot(check_point_list,np.mean(loss_per_iter_matrix,axis=1),linewidth=3)
    plt.title("{0} loss attack type {1}".format(algo, args.attack))
    plt.grid(visible=True,axis="y")
    plt.grid(visible=True,which='minor')
    plt.xlabel("iter")
    plt.ylabel("loss")

import cProfile,pstats



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

    font = {
        'weight': 'bold',
        'size': 16}
    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_loss_per_iter_matrix), axis=1),
             linewidth=3)
    plt.plot(iteration_list,
             np.mean(np.log(jopeq_loss_per_iter_matrix), axis=1), linewidth=3)
    plt.title("DLG Vanilla vs JoPEQ - Gradient Loss Metric")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("loss")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_mse_per_iter_matrix), axis=1),
             linewidth=3)
    plt.plot(iteration_list, np.mean(np.log(jopeq_mse_per_iter_matrix), axis=1),
             linewidth=3)
    plt.title("dlg vanilla MSE vs JoPEQ MSE")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("log(MSE)")

    plt.figure()
    plt.rc('font', **font)
    plt.plot(iteration_list, np.mean(np.log(dlg_ssim_per_iter_matrix), axis=1),
             linewidth=3)
    plt.plot(iteration_list,
             np.mean(np.log(jopeq_ssim_per_iter_matrix), axis=1), linewidth=3)
    plt.title("DLG vanilla vs JoPEQ SSIM")
    plt.grid(visible=True, axis="y")
    plt.grid(visible=True, which='minor')
    plt.xlabel("epoches")
    plt.ylabel("SSIM")

    plt.figure()
    plt.plot(iteration_list, np.mean(grads_norm_mat, axis=1), linewidth=3)

    plt.show()


def main():
    number_of_images = 1
    # image_number_list = [random.randrange(1, 1000, 1) for i in range(number_of_images)]

    # image_number_list = [3767]
    # epsilon_list = [0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001]
    epsilon_list = [0]
    check_point_list = [i for i in range(0, 400, 100)]
    model_number = 813665
    # run_dlg_idlg_tests(image_number_list,check_point_list,model_number,algo='DLG')
    # run_epsilon_dlg_idlg_tests(image_number_list,epsilon_list,algo='DLG')

    # run_dlg(30, learning_epoches=50, epsilon=0)
    # K = 25
    # print("image= {0}".format(K))
    # [0.1, 0.08, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    # imagen ids, epsilon list,
    epsilon_lst = [10, 33, 100, 333, 1000, 3333, 10000, 100000]
    bit_rate_lst = [4, 8, 16, 32]

    img_lst = list(range(30, 100))
    epsilon_lst = [333,3333,10000]
    bit_rate_lst = [8,16]
    iteration_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    iteration_lst = list(range(25))
    #img_lst = [16,17,18,19,20]
    # run_epsilon_dlg_idlg_tests(,[0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001],'DLG')
    # run_epsilon_dlg_idlg_tests(img_lst, epsilon_lst, bit_rate_lst=bit_rate_lst, algo=  'DLG')
    print("chosen images: {0}".format(img_lst))
    profiler = cProfile.Profile()
    profiler.enable()

    run_iteration_dlg_idlg_tests(img_lst, list(range(30)), 'DLG')

    # produce_image_pentas(img_lst, iteration_lst, epsilon_lst, bit_rate_lst)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('../output/cpu-export-data')
    # run_epsilon_dlg_idlg_tests([9],[0.0003,0.0001],'DLG')
    # run_epsilon_dlg_idlg_tests([9,10,11,12,13],[0.14,0.12,0.1,0.08,0.06,0.03,0.01,0.003,0.001,0.0003,0.0001,0.00001],'DLG')

    # run_dlg(K)
    # plt.show()
    pass


if __name__ == "__main__":
    main()




# plt.figure()
# font = {'weight': 'bold','size': 16}
# plt.rc('font', **font)
# bit_rate_lst = [2,4,8,16,32,64,128]
# with open("/Users/elad.sofer/src/Engineering Project/dlg/output/epsilon_mat_quant_onlyDLG.npy", "rb") as fd:
#     mat = np.load(fd)
#
#     plt.plot(bit_rate_lst[:6], np.mean(mat[:6],axis=1), 'r-*')
#     # plt.xscale("log")
#
#     plt.title("Quantization only DLG attack vs. compression rate")
#     plt.grid(visible=True,axis="y")
#     plt.grid(visible=True,which='minor')
#     plt.xlabel("compression rate (#bit number per level)")
#     plt.ylabel("loss")
#
#     bit_rate_lst = [2, 4, 8, 16, 32, 64, 128
#                     ]
# with open(
#         "/Users/elad.sofer/src/Engineering Project/dlg/output/epsilon_mat_DITH_QUANTDLG.npy",
#         "rb") as fd:
#     mat = np.load(fd)
#
#     plt.plot(epsilon_lst, np.mean(mat[:5], axis=1), '-*', linewidth=0.5)
#     plt.xscale("log")
#
#     plt.title("JoPEQ DLG attack vs. noise levels")
#     plt.grid(visible=True, axis="y")
#     plt.grid(visible=True, which='minor')
#     plt.xlabel("2/epsilon")
#     plt.ylabel("loss")

# plt.figure()
# font = {'weight': 'bold','size': 16}
# plt.rc('font', **font)
# epsilont_lst = [10,100,1000,10000,100000]
# bit_rate_lst = [2,4,8,16,32]
# with open("/Users/elad.sofer/src/Engineering Project/dlg/output/TOTAL_MATDLG.npy", "rb") as fd:
#     mat = pickle.load(fd)
#
# for k in range(0,mat.shape[0]):
#     plt.plot([2/e for e in epsilont_lst], np.mean(mat[k,:,:],axis=1), '-*')
#     plt.xscale("log")
#     plt.yscale("log")
#
#     plt.title("JoPEQ DLG attack vs. noise levels")
#     plt.grid(visible=True,axis="y")
#     plt.grid(visible=True,which='minor')
#     plt.xlabel("2/epsilon")
#     plt.ylabel("loss")
#
# plt.legend(["4compressionRate", "8compressionRate", "16compressionRate", "32compressionRate"])
pass