import torch
from torch import optim
from torchvision import datasets, transforms
from vision import LeNet, CNN, weights_init
from PIL import Image
from utils import label_to_onehot, cross_entropy_for_onehot
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import sys
tomer_path = r"C:\Users\tomer\Documents\Final_project_git\federated_learning_uveqfed_dlg\Federated-Learning-Natalie"
elad_path = r"/Users/elad.sofer/src/Engineering Project/federated_learning_uveqfed_dlg/Federated-Learning-Natalie"
sys.path.append(elad_path)
sys.path.append(tomer_path)

from federated_utils import PQclass
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"

def add_uveqFed(original_dy_dx, epsilon, bit_rate, args):
    noised_dy_dx = []
    args.epsilon = epsilon
    args.R = bit_rate
    noiser = PQclass(args)
    for g in original_dy_dx:
        if args.attack=='JOPEQ':
            output, dither = noiser(g)
            noised_dy_dx.append(output - dither)
            # output = noiser.apply_quantization(g)
            # noised_dy_dx.append(output)
        elif args.attack=="quantization":
            # quantization only
            output = noiser.apply_quantization(g)
            noised_dy_dx.append(output)
        else: # ppn only
            output = noiser.apply_privacy_noise(g)
            noised_dy_dx.append(output)

    return noised_dy_dx


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
class dlg_cls():
    def __init__(self,model=None, train_loader=None, test_loader=None, args=None, noise_func = lambda x, y, z, l: x):
        self.dst = getattr(datasets, args.dataset)("~/.torch", download=True)
        self.tp = transforms.ToTensor()
        self.tt = transforms.ToPILImage()
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.noise_func = noise_func


    def __call__(self, img_index, seed=1234,learning_epoches=0,read_grads= -1, epsilon=0, bit_rate=1,num_of_iterations=200):
        self.load_image(img_index)
        self.config_model(None,seed)
        self.train_model(learning_epoches)
        if (read_grads == -1):
            self.compute_gradients()
        else:
            self.load_model_and_gradients(read_grads)
        self.apply_noise(epsilon,bit_rate)
        return self.dlg(num_of_iterations=num_of_iterations)

    def load_image(self, img_index):
        self.img_index = img_index
        self.gt_data = self.tp(self.dst[img_index][0]).to(device)
        if len(self.args.image) > 1:
            self.gt_data = Image.open(self.args.image)
            self.gt_data = self.tp(self.gt_data).to(device)
        self.gt_data = self.gt_data.view(1, *self.gt_data.size())
        self.gt_label = torch.Tensor([self.dst[img_index][1]]).long().to(device)
        self.gt_label = self.gt_label.view(1, )
        self.gt_onehot_label = label_to_onehot(self.gt_label)
        return self.dst[self.img_index][0]

    def config_model(self,model=None,seed=1234):
        if model == None:
            self.model = LeNet().to(device)
        else:
            self.model = model
        torch.manual_seed(seed)
        self.model.apply(weights_init)
        self.model.to(device)
        self.criterion = cross_entropy_for_onehot
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self,learning_epoches=0):
        if (learning_epoches > 0):
            self.model.train_nn(
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch_num=learning_epoches,
                test_loader=self.test_loader)

            return self.model.test_nn(self.test_loader, self.criterion)
    def compute_gradients(self):
        self.pred = self.model(self.gt_data)
        y = self.criterion(self.pred, self.gt_onehot_label)
        self.dy_dx = torch.autograd.grad(y, self.model.parameters())
        self.original_dy_dx = self.dy_dx
        return self.dy_dx

    def load_model_and_gradients(self,read_grads):
        grad_checkpoint_address = "./fed-ler_checkpoints/grad/checkpoint{0}_{1}.pk".format(model_number, read_grads)
        global_checkpoint_address = "./fed-ler_checkpoints/global/checkpoint{0}_{1}.pk".format(model_number, read_grads)
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)

        global_model = torch.load(global_checkpoint_address)
        self.model = global_model
        # luckily the state dict is saved in exactly the same order as the gradients are so we can easily transfer them
        self.dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
        return self.dy_dx
    def apply_noise(self, epsilon, bit_rate, noise_func = None, args = None):
        if noise_func != None:
            self.noise_func = noise_func
        if args != None:
            self.args = args
        if (epsilon > 0 or self.args.attack=="quantization"):
            self.original_dy_dx = self.noise_func(list((_.detach().clone() for _ in self.dy_dx)), epsilon, bit_rate, self.args)
        else:
            self.original_dy_dx = self.dy_dx
    def dlg(self,num_of_iterations = 200):
        # generate dummy data and label
        dummy_data = torch.randn(self.gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn(self.gt_onehot_label.size()).to(device).requires_grad_(True)
        # plt.figure()
        # plt.imshow(tt(dummy_data[0].cpu()))

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

        # history = []
        current_loss = torch.Tensor([1])
        iters = 0
        MSE=0
        SSIM=0
        # while (iters < num_of_iterations):
        while (current_loss.item() > 0.00001 and iters < num_of_iterations):

            def closure():
                optimizer.zero_grad()

                dummy_pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                reconstructedIm = np.asarray(self.tt(dummy_data[0].cpu()))
                RecImShape = reconstructedIm.shape
                groundTruthIm = np.asarray(self.dst[self.img_index][0]).reshape((RecImShape[0], RecImShape[1], RecImShape[2]))
                MSE = mse(reconstructedIm,groundTruthIm)
                SSIM = ssim(reconstructedIm,groundTruthIm,channel_axis=2, multichannel=True)
                print(iters, "%.4f" % current_loss.item()," MSE {0:.4f}, SSIM {1:.4f}".format(MSE,SSIM))
                # history.append(self.tt(dummy_data[0].cpu()))
            iters = iters + 1

        self.final_image = self.tt(dummy_data[0].cpu())
        return current_loss.item(), MSE, SSIM


def run_dlg(img_index, model=None, train_loader=None, test_loader=None, noise_func = lambda x, y, z: x, learning_epoches = 0, epsilon=0.1, bit_rate=1,read_grads=-1,model_number=0):
    gt_data = tp(dst[img_index][0]).to(device)
    if len(args.image) > 1:
        gt_data = Image.open(args.image)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    #################### Model Configuration ####################

    model = LeNet().to(device)

    torch.manual_seed(1234)
    model.apply(weights_init)
    criterion = cross_entropy_for_onehot
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if (read_grads == -1):# run the original images
        #################### Train & Test ####################
        if (learning_epoches >0):
            model.train_nn(train_loader=train_loader, optimizer=optimizer, criterion=criterion,  epoch_num=learning_epoches,test_loader=test_loader)
            model.test_nn(test_loader,criterion)
        ######################################################
        # compute original gradient
        pred = model(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, model.parameters())
    else: # get the images from the fed-learn
        grad_checkpoint_address = "./fed-ler_checkpoints/grad/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        global_checkpoint_address = "./fed-ler_checkpoints/global/checkpoint{0}_{1}.pk".format(model_number,read_grads)
        fed_ler_grad_state_dict = torch.load(grad_checkpoint_address)


        global_model = torch.load(global_checkpoint_address)
        model =global_model
        # luckily the state dict is saved in exactly the same order as the gradients are so we can easily transfer them
        dy_dx = tuple([fed_ler_grad_state_dict[key] for key in fed_ler_grad_state_dict.keys()])
    #################### adding noise ####################
    if (epsilon > 0):
        original_dy_dx = noise_func(list((_.detach().clone() for _ in dy_dx)), epsilon, bit_rate)
    else:
        original_dy_dx = dy_dx

    #### adding noise!! ####
    #original_dy_dx = [w_layer + torch.normal(mean = 0, std= 0.01,size = w_layer.shape) for w_layer in original_dy_dx]
    #original_dy_dx = [w_layer+np.random.laplace(0,epsilon,w_layer.shape) for w_layer in original_dy_dx]
    ######################################################

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    plt.figure()
    plt.imshow(tt(dummy_data[0].cpu()))
    # plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


    history = []
    current_loss = torch.Tensor([1])
    iters = 0
    #for iters in range(num_of_iterations):
    # while (iters < num_of_iterations):
    while (current_loss.item()>0.00001 and iters < num_of_iterations):

        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
        iters = iters + 1

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(tt(dummy_data[0].cpu()))
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(dst[img_index][0])
    # plt.axis('off')

    # plt.figure(figsize=(12, 8))
    # for i in range(round(iters / 10)):
    #     plt.subplot(int(np.ceil(iters / 100)), 10, i + 1)
    #     plt.imshow(history[i])
    #     plt.title("iter=%d" % (i * 10))
    #     plt.axis('off')
    return current_loss.item()

# l = []
# for i in range(10):
#     l.append(test_image(img_index,learning_iterations=500+50*i))
# print(l)
#plt.hist([7 if (x>5) else x for x in l])
# plt.plot(l)
