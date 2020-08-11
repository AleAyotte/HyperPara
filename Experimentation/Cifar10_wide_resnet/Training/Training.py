"""
    @file:              Module.py
    @Author:            Alexandre Ayotte
    @Creation Date:     24/06/2020
    @Last modification: 02/07/2020

    The PreActResNet Model.
"""

from Experimentation.Cifar10_wide_resnet.Dataset.DataManager import dataset_to_loader
from Experimentation.Cifar10_wide_resnet.Model.Utils import init_weights, to_one_hot
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


class Trainer:
    def __init__(self, trainset, validset, lr_decay_step=2, tol=0.01, mixup_loss="bce", save_path=None,
                 pin_memory=False, device="cuda:0"):
        """
        The constructor of the trainer class. The trainer object can be use to create a specific model with mixup
        training.

        :param trainset: The training set
        :param validset: The validation set
        :param lr_decay_step: Number of learning rate decay step before ended the training. (Default=2)
        :param tol: Minimum difference between the best and the current loss to consider that there is an improvement.
                    (Default=0.01)
        :param mixup_loss: The loss that will be use during mixup epoch. (Default="bce")
        :param save_path: Directory path where the checkpoints will be write during the training
        :param pin_memory: If true, activate the pin_memory option of the dataloader. (Default=False)
        :param device: The device on which the training will be done. (Default="cuda:0", first GPU)
        :return: A trainer object to train a given model.
        """

        assert mixup_loss == "ce" or mixup_loss == "bce", \
            "The loss option are ce (cross entropy) or bce (binary cross entropy)"

        self.model = None

        self.mixup_loss = mixup_loss
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.soft = torch.nn.Softmax(dim=-1)

        self.trainset = trainset
        self.validset = validset
        self.lr_decay_step = lr_decay_step
        self.tol = tol

        self.save_path = save_path
        self.pin_memory = pin_memory
        self.device = device

    def fit(self, model, num_epoch=200, batch_size=150, learning_rate=0.01, momentum=0.9, l2=1e-4, t_0=200,
            eta_min=1e-4, grad_clip=0, mode="standard", warm_up_epoch=0, retrain=False, device="cuda:0", verbose=True):
        """
        Train a given model on the trainer dataset using the givens hyperparameters.

        :param model: The model to train.
        :param num_epoch: Maximum number of epoch during the training. (Default=200)
        :param batch_size: The batch size that will be used during the training. (Default=150)
        :param learning_rate: Start learning rate of the SGD optimizer. (Default=0.1)
        :param momentum: Momentum that will be used by the SGD optimizer. (Default=0.9)
        :param l2: L2 regularization coefficient.
        :param t_0: Number of epoch before the first restart. (Default=200)
        :param eta_min: Minimum value of the learning rate. (Default=1e-4)
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient. (Default=0)
        :param mode: The training type: Option: Standard training (No mixup) (Default)
                                                Mixup (Standard manifold mixup)
                                                AdaMixup (Adaptative mixup)
                                                ManifoldAdaMixup (Manifold Adaptative mixup)
        :param warm_up_epoch: Number of iteration before activating mixup. (Default=True)
        :param retrain: If false, the weights of the model will initialize. (Default=False)
        :param device: The device on which the training will be done. (Default="cuda:0", first GPU)
        :param verbose: If true, show the progress of the training. (Default=True)
        """
        # Indicator for early stopping
        best_accuracy = 0
        best_epoch = -1
        current_mode = "Standard"

        # We get the appropriate loss because mixup loss will always be bigger than standard loss.
        last_saved_loss = float("inf")

        # Initialization of the model.
        self.device = device
        self.model = model.to(device)
        self.model.configure_mixup_module(batch_size, device)

        if retrain:
            start_epoch, last_saved_loss, best_accuracy = self.model.restore(self.save_path)
        else:
            self.model.apply(init_weights)
            start_epoch = 0

        # Initialization of the dataloader
        train_loader = dataset_to_loader(self.trainset, batch_size, shuffle=True, pin_memory=self.pin_memory)
        valid_loader = dataset_to_loader(self.validset, batch_size, shuffle=False, pin_memory=self.pin_memory)

        # Initialization of the optimizer and the scheduler
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=l2,
            nesterov=True)

        n_iters = len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0*n_iters,
            T_mult=1,
            eta_min=eta_min)
        scheduler.step(start_epoch*n_iters)

        # Go in training mode to activate mixup module
        self.model.train()

        with tqdm(total=num_epoch, initial=start_epoch, disable=(not verbose)) as t:
            for epoch in range(start_epoch, num_epoch):

                _grad_clip = 0 if epoch > num_epoch / 2 else grad_clip
                current_mode = mode if warm_up_epoch <= epoch else current_mode

                # We make a training epoch
                if current_mode == "Mixup":
                    training_loss = self.mixup_epoch(train_loader, optimizer, scheduler, _grad_clip)
                else:
                    training_loss = self.standard_epoch(train_loader, optimizer, scheduler, _grad_clip)

                self.model.eval()
                current_accuracy, val_loss = self.accuracy(dt_loader=valid_loader, get_loss=True)
                self.model.train()

                # ------------------------------------------------------------------------------------------
                #                                   EARLY STOPPING PART
                # ------------------------------------------------------------------------------------------

                if (val_loss < last_saved_loss and current_accuracy >= best_accuracy) or \
                        (val_loss < last_saved_loss*(1+self.tol) and current_accuracy > best_accuracy):
                    self.save_checkpoint(epoch, val_loss, current_accuracy)
                    best_accuracy = current_accuracy
                    last_saved_loss = val_loss
                    best_epoch = epoch

                if verbose:
                    t.postfix = "train loss: {:.4f}, val loss: {:.4f}, val acc: {:.2f}%, best acc: {:.2f}%, " \
                                "best epoch: {}, epoch type: {}".format(
                                 training_loss, val_loss, current_accuracy * 100, best_accuracy * 100, best_epoch + 1,
                                 current_mode)
                    t.update()
        self.model.restore(self.save_path)

    def mixup_criterion(self, pred, target, mixup_position=None, lamb=None, permut=None):
        """
        Transform target into one hot vector and apply mixup on it

        :param pred: A maxtrix of the prediction of the model.
        :param target: Vector of the ground truth
        :param mixup_position: Position of the module in the self.conv sequantial container
        :param lamb: The mixing paramater that has been used to produce the mixup during the foward pass
        :param permut: A numpy array that indicate which images has been shuffle during the foward pass
        :return: The mixup loss as torch tensor
        """

        if mixup_position is not None:
            if self.mixup_loss == "bce":
                hot_target = to_one_hot(target, self.model.num_classes, self.device)
                mixed_target = lamb*hot_target + (1-lamb)*hot_target[permut]
                return self.bce_loss(self.soft(pred), mixed_target)

            else:
                return lamb*self.ce_loss(pred, target) + (1-lamb)*self.ce_loss(pred, target[permut])
        else:
            return self.ce_loss(pred, target)

    def init_mixup(self):
        """
        Initialize the mixup modules

        :return: Index of the layer where the mixup is done
        """

        if len(self.model.mixup_index) > 0:

            # We select randomly a mixup module and we activate him
            layer = self.model.mixup_index[random.randint(0, len(self.model.mixup_index) - 1)]
            lamb, permut = self.model.conv_layers[layer].sample()

            # Return the index of the next mixup layer
            return layer, lamb, permut

        else:
            return None, None, None

    def mixup_epoch(self, train_loader, optimizer, scheduler, grad_clip):
        """
        Make a manifold mixup epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizer: The torch optimizer that will used to train the model.
        :param scheduler: The learning rate scheduler that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss
        """

        sum_loss = 0
        n_iters = len(train_loader)

        for step, data in enumerate(train_loader, 0):
            features, labels = data[0].to(self.device), data[1].to(self.device)
            features, labels = torch.autograd.Variable(features), torch.autograd.Variable(labels)

            optimizer.zero_grad()

            # Mixup activation
            mixup_layer, lamb, permut = self.init_mixup()

            # training step
            pred = self.model(features)
            loss = self.mixup_criterion(pred, labels, mixup_layer, lamb, permut)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            # Save the loss
            sum_loss += loss

            if mixup_layer is not None:
                self.model.conv_layers[mixup_layer].enable = False

        return sum_loss.item() / n_iters

    def standard_epoch(self, train_loader, optimizer, scheduler, grad_clip):
        """
        Make a standard training epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizer: The torch optimizer that will used to train the model.
        :param scheduler: The learning rate scheduler that will be used at each iteration.
        :param grad_clip: Max norm of the gradient. If 0, no clipping will be applied on the gradient.
        :return: The average training loss
        """

        sum_loss = 0
        n_iters = len(train_loader)

        for step, data in enumerate(train_loader, 0):
            features, labels = data[0].to(self.device), data[1].to(self.device)
            features, labels = torch.autograd.Variable(features), torch.autograd.Variable(labels)

            optimizer.zero_grad()

            # training step
            pred = self.model(features)
            loss = self.ce_loss(pred, labels)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            # Save the loss
            sum_loss += loss

        return sum_loss.item() / n_iters

    def accuracy(self, dt_loader, get_loss=False):

        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data
        :param get_loss: Return also the loss if True
        :return: The accuracy of the model and the average loss if get_loss == True
        """

        accuracy = np.array([])
        loss_vect = np.array([])

        for data in dt_loader:
            features, labels = data[0].to(self.device), data[1]

            with torch.no_grad():
                out = torch.Tensor.cpu(self.model(features))
                loss = self.ce_loss(out, labels)

            pred = np.argmax(out.numpy(), axis=1)
            accuracy = np.append(accuracy, np.where(pred == labels.numpy(), 1, 0).mean())
            loss_vect = np.append(loss_vect, loss)

        if get_loss:
            return accuracy.mean(), loss_vect.mean().item()
        else:
            return accuracy.mean()

    def score(self, testset=None, batch_size=150):

        """
        Compute the accuracy of the model on a given test dataset

        :param testset: A torch dataset which contain our test data points and labels
        :param batch_size: The batch_size that will be use to create the data loader. (Default=150)
        :return: The accuracy of the model.
        """

        test_loader = dataset_to_loader(testset, batch_size, shuffle=False)

        self.model.eval()

        return self.accuracy(dt_loader=test_loader)

    def save_checkpoint(self, epoch, loss, accuracy):

        """
        Save the model and his at a the current state if the self.path is not None.

        :param epoch: Current epoch of the training
        :param loss: Current loss of the training
        :param accuracy: Current validation accuracy
        """

        if self.save_path is not None:
            torch.save({"epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "loss": loss,
                        "accuracy": accuracy},
                       self.save_path)
