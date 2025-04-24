import torch
from ROLANN import ROLANN
from Cliente import Cliente
from Coordinador import Coordinador
from datasets import get_dataset
from torch.utils.data import random_split

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset

def non_iid_dirichlet_partition(dataset, num_clients, alpha):

    # Extraemos las etiquetas del dataset como array de NumPy
    targets = np.asarray(dataset.targets)

    # Obtenemos el número de clases
    num_classes = len(np.unique(targets))

    # Lista de listas de índices para cada clase
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    # Inicializamos una lista vacía por cliente
    client_indices = [[] for _ in range(num_clients)]

    # Distribuimos cada clase entre los clientes
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])  # evitar bloques consecutivos
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Transforma proporciones a enteros para saber cuántos ejemplos asignar a cada cliente
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]

        # Toma el array de indices de la clase c y lo divide entre los clientes en funcion de las proporciones
        splits = np.split(class_indices[c], proportions)

        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist()) # Agrega los indices (posicion imagen) a la lista del cliente correspondiente

    return client_indices

  
def non_iid_class_less_partition(dataset, num_clients, classes_per_client=2):

    # Extraemos las etiquetas del dataset como array de NumPy
    targets = np.asarray(dataset.targets)
    num_classes = len(np.unique(targets))

    # Creamos un diccionario que guarda los índices de cada clase
    class_indices = {}
    for i in range(num_classes):
        indices = np.where(targets == i)[0]
        class_indices[i] = indices.tolist()

    # Inicializamos una lista vacía por cliente para guardar sus índices
    client_indices = [[] for i in range(num_clients)]

    # Asignamos a cada cliente un subconjunto aleatorio de clases
    for client_id in range(num_clients):

        # Determina que clases le tocan al cliente (al azar)
        classes = np.random.choice(num_classes, classes_per_client, replace=False) # classes es un array de enteros con las clases

        for cls in classes:
            # Verificamos que haya suficientes muestras disponibles en la clase
            if len(class_indices[cls]) >= num_clients:
                take = len(class_indices[cls]) // num_clients # Tomamos un número proporcional de muestras
            else:
                take = min(len(class_indices[cls]), 1)  # Toma el mínimo entre el número de muestras disponibles y 1 (puede ser 0)

            # Seleccionamos los índices para este cliente y actualizamos las listas
            selected = class_indices[cls][:take]
            client_indices[client_id].extend(selected)
            class_indices[cls] = class_indices[cls][take:]

    return client_indices


def plot_client_distributions(client_indices, dataset, title="Distribución de clases por cliente"):
    num_clients = len(client_indices)
    class_counts = []
    targets = np.array(dataset.targets)
    for indices in client_indices:
        labels = targets[indices]
        count = Counter(labels)
        class_counts.append([count.get(i, 0) for i in range(10)])

    data = np.array(class_counts)
    for i in range(10):
        plt.bar(range(num_clients), data[:, i], bottom=np.sum(data[:, :i], axis=1), label=f'Clase {i}')

    plt.xlabel("Cliente")
    plt.ylabel("Número de muestras")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Dataset
    dataset = 1  # 0: MNIST, 1: CIFAR10

    try:
        # Obtenemos el conjunto de imagenes a repartir y los loaders para la evaluación
        train_imgs, train_loader, test_loader = get_dataset(dataset)
    except ValueError as e:
        print(f"Error al cargar el dataset: {e}")
        exit()

    # Configuramos la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creamos un Coordinador
    coordinador = Coordinador(ROLANN(num_classes=10), device)

    # Numero de clientes que queremos crear
    num_clientes = 12
    clientes = [] # Lista de clientes

    # Cada cliente tendrá un subconjunto del dataset, su propia ResNet y su propio ROLANN

    partition_type = "class_less"  # "iid", "dirichlet", "class_less"

    if partition_type == "iid":

        # Creamos los subconjuntos de datos para cada cliente
        dataset_size = len(train_imgs)
        print(f"Dataset size: {dataset_size}")

        batch_size = [dataset_size // num_clientes] * num_clientes # Creamos una lista con el tamaño del batch para cada cliente
        resto = dataset_size % num_clientes
        print(f"Batch size: {batch_size}")
        print(f"Resto: {resto}")

        # Si la división no es exacta, ajustamos el tamaño del último cliente
        if resto > 0:
            for i in range(resto):
                batch_size[i] += 1

        # Comprobamos que la suma de los tamaños de los batches sea igual al tamaño del dataset
        print("Longitudes por cliente:", batch_size)
        print("Suma de longitudes:", sum(batch_size))

        if sum(batch_size) != dataset_size:
            print("Error: la suma de los tamaños de los batches no es igual al tamaño del dataset")
            exit()

        dataset_dividido = random_split(train_imgs, batch_size)
        client_subsets = [Subset(train_imgs, s.indices) for s in dataset_dividido]
        client_indices = [s.indices for s in dataset_dividido]
        plot_client_distributions(client_indices, train_imgs, "Distribución IID")

    elif partition_type == "dirichlet":
        client_indices = non_iid_dirichlet_partition(train_imgs, num_clientes, alpha=0.3)

        # Crear un subconjunto de datos para cada cliente en función de los índices
        client_subsets = [Subset(train_imgs, indices) for indices in client_indices] 
        plot_client_distributions(client_indices, train_imgs, "Dirichlet α=0.3")

    elif partition_type == "class_less":
        client_indices = non_iid_class_less_partition(train_imgs, num_clientes, classes_per_client=3)

        # Crear un subconjunto de datos para cada cliente en función de los índices
        client_subsets = [Subset(train_imgs, indices) for indices in client_indices]
        plot_client_distributions(client_indices, train_imgs, "Class Less (sin todas las clases)")

    else:
        print("Tipo de partición no soportado. Usa 'iid', 'dirichlet' o 'class_less'")
        exit()

    
    # Creamos los clientes
    for i in range(num_clientes):
        print(f"Cliente {i}: Inicializando...")
        clientes.append(Cliente(ROLANN(num_classes=10), client_subsets[i], device))



    global_M_list = []  # Aquí se guardarán las actualizaciones de M de cada cliente
    global_US_list = [] # Aquí se guardarán las actualizaciones de US de cada cliente

    # Entrenamos a todos los clientes y recopilamos sus actualizaciones
    for i, cliente in enumerate(clientes):

        print(f"Entrenando Cliente {i}...")
        cliente.training()  # Entrena localmente al cliente

        local_M, local_US = cliente.aggregate_parcial()  # Extrae las matrices locales del cliente

        global_M_list.append(local_M)
        global_US_list.append(local_US)
        print(f"Cliente {i} entrenado.")

    # Ahora, el Coordinador realiza la agregación global con todas las actualizaciones
    coordinador.recolect_parcial(global_M_list, global_US_list)
    print("Agregación global completada.")


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

    print("Evaluando el modelo global...")
    train_acc = evaluate(coordinador.rolann, coordinador.resnet, train_loader)
    test_acc = evaluate(coordinador.rolann, coordinador.resnet, test_loader)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
