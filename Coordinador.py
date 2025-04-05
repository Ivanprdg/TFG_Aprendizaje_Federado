import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Coordinador:

    def __init__(self, rolann, device):

        self.rolann = rolann
        self.device = device

        self.M_glb = []  # Matriz M global acumulada para cada clase
        self.U_glb = []  # Matriz U global acumulada para cada clase
        self.S_glb = []  # Matriz S global acumulada para cada clase


        # ResNet preentrenada y congelada
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # resnet propia
        self.resnet.fc = nn.Identity()  # Remplazamos la capa final para extraer características

        for param in self.resnet.parameters():
            param.requires_grad = False  # Congelamos la ResNet

        self.resnet.to(self.device)
        self.resnet.eval()



    def recolect_parcial(self, M_list, US_list):
        """
        Recolecta las matrices M y US de cada cliente y las agrega para formar el modelo global.
        """
        # Number of classes
        nclasses = len(M_list[0])
        init = False

        # For each class the results of each client are aggregated    
        for c in range(0, nclasses):

            if (not self.M_glb) or init:            
                init = True
                M  = M_list[0][c]
                US = US_list[0][c]
                M_rest  = [item[c] for item in M_list[1:]]
                US_rest = [item[c] for item in US_list[1:]]
            else:
                M = self.M_glb[c]
                US = self.U_glb[c] @ np.diag(self.S_glb[c])
                M_rest  = [item[c] for item in M_list[:]]
                US_rest = [item[c] for item in US_list[:]]

            # Aggregation of M and US
            for M_k, US_k in zip(M_rest, US_rest):
                M = M + M_k
                
                # Conversión robusta de US_k
                if isinstance(US_k, torch.Tensor):
                    US_k_np = US_k.cpu().numpy()
                else:
                    US_k_np = US_k
                
                # Conversión robusta de US
                if isinstance(US, torch.Tensor):
                    US_np = US.cpu().numpy()
                else:
                    US_np = US

                U, S, _ = sp.linalg.svd(np.concatenate((US_k_np, US_np), axis=1), full_matrices=False)
                US = U @ np.diag(S)
            
            # Save contents
            if init:
                self.M_glb.append(M)
                self.U_glb.append(U)
                self.S_glb.append(S)
            else:
                self.M_glb[c] = M
                self.U_glb[c] = U
                self.S_glb[c] = S

        self.actualizar_modelo_global(self.M_glb, self.U_glb, self.S_glb)



    def actualizar_modelo_global(self, mg_list, ug_list, sg_list):
        """
        Actualiza el modelo global de ROLANN con las matrices globales calculadas.
        """        
        mg_tensor_list = [m if isinstance(m, torch.Tensor) else torch.tensor(m, device=self.device) for m in mg_list]
        ug_tensor_list = [u if isinstance(u, torch.Tensor) else torch.from_numpy(u).to(self.device) for u in ug_list]
        sg_tensor_list = [s if isinstance(s, torch.Tensor) else torch.from_numpy(s).to(self.device) for s in sg_list]
        
        # Asignamos las listas directamente al modelo global
        self.rolann.mg = mg_tensor_list
        self.rolann.ug = ug_tensor_list
        self.rolann.sg = sg_tensor_list
        
        # Recalcula los pesos globales en base a las nuevas matrices agregadas
        self.rolann._calculate_weights()

