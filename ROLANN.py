# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar Fontenla & Alejandro Dopico
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ROLANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lamb: float = 0.01,
        sparse: bool = False,
        activation: str = "logs",
    ):
        super(ROLANN, self).__init__()

        self.num_classes = num_classes
        self.lamb = lamb  # Regularization hyperparameter

        if activation == "logs":  # Logistic activation functions
            self.f = torch.sigmoid
            self.finv = lambda x: torch.log(x / (1 - x))
            self.fderiv = lambda x: x * (1 - x)
        elif activation == "rel":  # ReLU activation functions
            self.f = F.relu
            self.finv = lambda x: torch.log(x)
            self.fderiv = lambda x: (x > 0).float()
        elif activation == "lin":  # Linear activation functions
            self.f = lambda x: x
            self.finv = lambda x: x
            self.fderiv = lambda x: torch.ones_like(x)

        self.w = []

        self.m = None
        self.u = None
        self.s = None

        self.mg = []
        self.ug = []
        self.sg = []

        self.sparse = sparse

    def update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        results = [self._update_weights(X, d[:, i]) for i in range(self.num_classes)]

        ml, ul, sl = zip(*results)

        self.m = torch.stack(ml, dim=0)
        self.u = torch.stack(ul, dim=0)
        self.s = torch.stack(sl, dim=0)

    def _update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)  # Number of data points (n)

        # The bias is included as the first input (first row)
        ones = torch.ones((1, n), device=X.device)

        xp = torch.cat((ones, X), dim=0)

        # Inverse of the neural function
        f_d = self.finv(d)

        # Derivative of the neural function
        derf = self.fderiv(f_d)

        if self.sparse:
            F_sparse = torch.diag(derf)

            H = torch.matmul(xp, F_sparse)

            U, S, _ = torch.linalg.svd(H, full_matrices=False)

            M = torch.matmul(
                xp, torch.matmul(F_sparse, torch.matmul(F_sparse, f_d.T))
            ).flatten()
        else:
            # Diagonal matrix
            F = torch.diag(derf)

            H = torch.matmul(xp, F)

            U, S, _ = torch.linalg.svd(H, full_matrices=False)

            M = torch.matmul(xp, torch.matmul(F, torch.matmul(F, f_d)))

        return M, U, S

    def forward(self, X: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)

        n_outputs = len(self.w)

        # Neural Network Simulation
        ones = torch.ones((1, n), device=X.device)
        xp = torch.cat((ones, X), dim=0)

        y_hat = torch.empty((n_outputs, n), device=X.device)

        for i in range(n_outputs):
            w_tmp = self.w[i].permute(
                *torch.arange(self.w[i].ndim - 1, -1, -1)
            )  # Trasposing

            y_hat[i] = self.f(torch.matmul(w_tmp, xp))

        return torch.transpose(y_hat, 0, 1)

    def _aggregate_parcial(self) -> None:
        for i in range(self.num_classes):
            if i >= len(self.mg):
                # Initialization using the first element of the list
                M = self.m[i]
                U = self.u[i]
                S = self.s[i]

                self.mg.append(M)
                self.ug.append(U)
                self.sg.append(S)

            else:
                M = self.mg[i]
                m_k = self.m[i]
                s_k = self.s[i]
                u_k = self.u[i]

                US = torch.matmul(self.ug[i], torch.diag(self.sg[i]))

                # Aggregation of M and US
                M = M + m_k
                us_k = torch.matmul(u_k, torch.diag(s_k))
                concatenated = torch.cat((us_k, US), dim=1)
                U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

                self.mg[i] = M
                self.ug[i] = U
                self.sg[i] = S

    def _calculate_weights(
        self,
    ) -> None:
        if not self.mg or not self.ug or not self.sg:
            return None

        for i in range(self.num_classes):
            M = self.mg[i]
            U = self.ug[i]
            S = self.sg[i]

            if self.sparse:
                I_ones = torch.ones(S.size())
                I_ones_size = list(I_ones.shape)[0]
                I_sparse = torch.sparse.spdiags(
                    I_ones,
                    torch.tensor(0),
                    (I_ones_size, I_ones_size),
                    layout=torch.sparse_csr,
                )
                S_size = list(S.shape)[0]
                S_sparse = torch.sparse.spdiags(
                    S, torch.tensor(0), (S_size, S_size), layout=torch.sparse_csr
                )

                aux = (
                    S_sparse.to_dense() * S_sparse.to_dense()
                    + self.lamb * I_sparse.to_dense()
                )
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
                w = torch.matmul(
                    U, torch.matmul(torch.linalg.pinv(aux), torch.matmul(U.T, M))
                )
            else:
                diag_elements = 1 / (
                    S * S + self.lamb * torch.ones_like(S, device=S.device)
                )
                diag_matrix = torch.diag(diag_elements)
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
                w = torch.matmul(U, torch.matmul(diag_matrix, torch.matmul(U.T, M)))

            if i >= len(self.w):
                # Append optimal weights
                self.w.append(w)
            else:
                self.w[i] = w

    def aggregate_update(self, X: Tensor, d: Tensor) -> None:
        self.update_weights(X, d)  # The new M and US are calculated
        self._aggregate_parcial()  # New M and US added to old (global) ones
        self._calculate_weights()  # The weights are calculated with the new

if __name__ == "__main__":

    # Dataset
    dataset = 0  # 0: MNIST, 1: CIFAR10

    # Configuramos la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 0:

        # Load MNIST dataset
        transform_mnist = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()]
        )
        mnist_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        mnist_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )

        # Cargamos los datos para el entrenamiento y test
        train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

    elif dataset == 1:

        # Load CIFAR10 dataset
        transform_cifar = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.ToTensor()]
        )

        cifar_train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_cifar
        )
        cifar_test = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_cifar
        )

        # Cargamos los datos para el entrenamiento y test
        train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
    else:
        raise ValueError("Invalid dataset value.")

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # Modelo ResNet18 preentrenado
    resnet.fc = nn.Identity() # Sustituimos la capa fc por una capa identidad

    # Congelamos la ResNet si queremos que no se entrene más:
    for param in resnet.parameters():
        param.requires_grad = False

    rolann = ROLANN(num_classes=10)

    resnet.to(device)  # Se sube la RESNET a la GPU
    rolann.to(device)  # Se sube la ROLANN a la GPU


    # Training
    for x, y in tqdm(train_loader):

        x = x.to(device) # Subimos los datos a la GPU
        y = y.to(device) # Subimos las etiquetas a la GPU

        with torch.no_grad():
            caracterisiticas = resnet(x)

        label = torch.nn.functional.one_hot(y, num_classes=10) * 0.9 + 0.05
        rolann.aggregate_update(caracterisiticas, label)

    def evaluate(model_rolann, model_resnet, loader): # Añadimos el modelo de ResNet18
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:

                x = x.to(device) # Subimos los datos a la GPU
                y = y.to(device) # Subimos las etiquetas a la GPU

                caracterisiticas = model_resnet(x) # Obtenemos las características de la ResNet18
                preds = model_rolann(caracterisiticas) # Obtenemos las predicciones de la ROLANNs

                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / total

    train_acc = evaluate(rolann, resnet, train_loader)
    test_acc = evaluate(rolann, resnet, test_loader)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
