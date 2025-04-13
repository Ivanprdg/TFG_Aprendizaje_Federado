import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

class Cliente:
    def __init__(self, rolann, dataset, device):

        self.rolann = rolann # Instancia de la clase ROLANN
        self.device = device # Dispositivo (CPU o GPU) donde se ejecutará el cliente
        self.loader = DataLoader(dataset, batch_size=128, shuffle=True) # dataset local

        # Cada cliente crea su propia ResNet preentrenada y congelada
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # resnet propia
        self.resnet.fc = nn.Identity()  # Remplazamos la capa final para extraer características

        for param in self.resnet.parameters():
            param.requires_grad = False  # Congelamos la ResNet

        self.resnet.eval()
        self.rolann.to(self.device)  # Aseguramos que ROLANN esté en el mismo dispositivo

    
    def training(self):
        """
        Recorre el dataset local, extrae las características usando la ResNet propia y
        actualiza la capa ROLANN
        """
        self.resnet.to(self.device) # Mover al training y pasar a cpu al terminar training

        for x, y in tqdm(self.loader):

            x = x.to(self.device)

            with torch.no_grad():
                features = self.resnet(x)  # Extraemos características locales

            # Convertimos las etiquetas a one-hot para que coincidan con el número de clases
            label = (torch.nn.functional.one_hot(y, num_classes=10) * 0.9 + 0.05).to(self.device)
            self.rolann.aggregate_update(features, label)

        # Movemos la resnet a cpu
        self.resnet.to("cpu")

    def aggregate_parcial(self):
        """
        Devuelve las matrices acumuladas M y US para cada clase
        """
        # Devuelve las matrices acumuladas M y US para cada clase
        local_M = self.rolann.mg
        local_US = [torch.matmul(self.rolann.ug[i], torch.diag(self.rolann.sg[i].clone().detach())) for i in range(self.rolann.num_classes)]

        return local_M, local_US
