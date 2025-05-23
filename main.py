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

import time
import tenseal as ts

# Reparticion de datos no-IID dirichlet, cada cliente tiene al menos una muestra de cada clase
def non_iid_dirichlet_partition(dataset, num_clients, alpha): 

    targets = np.asarray(dataset.targets) # Convertimos a array de NumPy
    num_classes = len(np.unique(targets)) # Obtenemos el número de clases
    
    # Creamos una lista de indices para cada clase
    class_indices = [np.where(targets == c)[0].tolist() for c in range(num_classes)]
    # Inicializamos la lista de indices por cliente
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes): # Para cada clase
        idxs = class_indices[c] # Obtenemos los indices de la clase c
        np.random.shuffle(idxs) # Barajamos los indices de la clase c para luego cortar bloques con datos mezclados
        N = len(idxs) # Obtenemos el numero de muestras de la clase c

        # Muestreamos proporciones Dirichlet
        props = np.random.dirichlet(alpha * np.ones(num_clients))
        raw = props * N # Obtenemos las proporciones de cada cliente

        # Aseguramos que cada cliente tiene al menos una muestra de la clase c
        counts = np.floor(raw).astype(int)
        counts[counts == 0] = 1

        # Ajustamos para que sum(counts) == N
        residual = raw - np.floor(raw)
        total = counts.sum()
        # si faltan muestras, vamos añadiendo a los mayores residuals
        while total < N:
            i = np.argmax(residual)
            counts[i] += 1
            residual[i] = 0
            total += 1
        # si sobran, quitamos de los menores residuals (pero sin caer bajo 1)
        while total > N:

            validos = counts > 1 # nos aseguramos de que no caiga por debajo de 1

            # entre ellos, buscamos el más pequeño residual
            j = np.argmin(np.where(validos, residual, np.inf))
            counts[j] -= 1 # quitamos una muestra al cliente j
            residual[j] = 1  # lo marcamos como ya quitado
            total -= 1 # restamos una muestra al total

        # A partir de counts, hacemos splits “manuales”
        start = 0
        for client_id, cnt in enumerate(counts):
            if cnt > 0: # si el cliente tiene muestras de la clase c
                split = idxs[start:start + cnt] # Obtenemos los indices de la clase c para el cliente client_id
                client_indices[client_id].extend(split) # Añadimos los indices al cliente
                start += cnt # Aumentamos el inicio para la siguiente clase

    return client_indices


# Reparto no-IID “class-less”: Los clientes tienen al menos una clase, pero no todas las clases
def non_iid_class_less_partition(dataset, num_clients, clases_privadas, alpha=0.5):

    targets = np.asarray(dataset.targets) # Convertimos a array de NumPy
    num_classes = len(np.unique(targets)) # Obtenemos el número de clases

    # Creamos una lista de indices para cada clase
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    all_classes = set(range(num_classes)) # Aqui guardamos todas las clases


    # Determinar cuantas clases serán privadas
    num_private = int(num_classes * clases_privadas) # Porcentaje de clases privadas
    num_private = min(num_private, num_classes) # Aseguramos que no sea mayor que el número de clases

    private_classes = list(np.random.choice(list(all_classes), size=num_private, replace=False)) # Elegimos las clases privadas al azar
    print(f"Clases privadas: {private_classes}")
    shared_classes = list(all_classes - set(private_classes)) # Clases compartidas entre clientes

    client_classes = [set() for _ in range(num_clients)] # Inicializamos un conjunto de clases para cada cliente

    # Asignar clases privadas en “round-robin” sobre los clientes
    for idx, c in enumerate(private_classes):
        client = idx % num_clients
        client_classes[client].add(c)

    # Asignar clases compartidas aleatoriamente a los clientes
    if shared_classes:
        for i in range(num_clients):
            k = np.random.randint(1, len(shared_classes) + 1)
            picks = set(np.random.choice(shared_classes, size=k, replace=False))
            client_classes[i].update(picks)

    # Si la union detecta que union != num_classes asignamos cada clase faltante
    union = set().union(*client_classes)
    missing = set(shared_classes) - union # Clases que faltan por asignar
    for clase in missing:
        # elige un cliente que aun no tenga todas las clases
        candidatos = [i for i, s in enumerate(client_classes) if len(s) < num_classes-1]
        elegido = np.random.choice(candidatos)
        client_classes[elegido].add(clase)

    # Evitar que algun cliente tenga todas las clases
    for i, s in enumerate(client_classes):
        if len(s) == num_classes: # si tiene todas las clases
            # elige una clase compartida (otro cliente tambien la tiene)
            compartidas = [c for c in s if sum(1 for s2 in client_classes if c in s2) > 1] # El if comprueba que la clase esta en mas de un cliente
            drop = np.random.choice(compartidas)
            s.remove(drop)

    # Reparto de indices por Dirichlet
    client_indices = [[] for i in range(num_clients)]
    for c, idxs in enumerate(class_indices):
        np.random.shuffle(idxs)
        clientes = [i for i, s in enumerate(client_classes) if c in s] 
        props = np.random.dirichlet(alpha * np.ones(len(clientes)))
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for client_id, split in zip(clientes, splits): # asignamos los indices a los clientes
            client_indices[client_id].extend(split.tolist())

    return client_indices


def plot_client_distributions(client_indices, dataset, title="Distribución de clases por cliente"):

    num_clients = len(client_indices)
    targets = np.array(list(dataset.targets))

    # Contar por cliente y clase
    class_counts = []
    num_classes = len(np.unique(targets))
    for indices in client_indices:
        labels = targets[indices]
        cnt = Counter(labels)
        class_counts.append([cnt.get(c, 0) for c in range(num_classes)])
    data = np.array(class_counts)  # shape (num_clients, num_classes)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Barras apiladas
    bottoms = np.zeros(num_clients, dtype=int)
    for cls in range(num_classes):
        ax.bar(
            range(num_clients),
            data[:, cls],
            bottom=bottoms,
            label=f"Clase {cls}"
        )
        bottoms += data[:, cls]

    # Etiquetas del eje X solo en enteros
    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f"Cliente {i+1}" for i in range(num_clients)])

    # Ajustar límites
    ax.set_xlim(-0.5, num_clients - 0.5)

    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    ax.set_xlabel("Cliente")
    ax.set_ylabel("Número de muestras")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def main():
    # Dataset
    dataset = 0  # 0: MNIST, 1: CIFAR10

    try:
        # Obtenemos el conjunto de imagenes a repartir y los loaders para la evaluación
        train_imgs, train_loader, test_loader = get_dataset(dataset)
    except ValueError as e:
        print(f"Error al cargar el dataset: {e}")
        exit()

    # Configuramos la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Numero de clientes que queremos crear
    num_clientes = 8
    clientes = [] # Lista de clientes

    encrypted = False # Si queremos usar cifrado o no

    if encrypted:
        # Creamos el contexto del encriptador
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40

        ctx_secret_key = ctx.serialize(save_secret_key=True)  # incluye secret key 
        ctx.make_context_public() # aqui lo que se hace es hacer público el contexto, pero no la clave secreta
        ctx_no_secret_key = ctx.serialize() # aqui se guarda el contexto público pero no la clave secreta

    # Creamos un Coordinador
    if encrypted:
        ctx = ts.context_from(ctx_no_secret_key) # no contiene la clave secreta
    else:
        ctx = None
    coordinador = Coordinador(ROLANN(num_classes=10, encrypted=encrypted, context=ctx), device, num_clients=num_clientes, broker="localhost", port=1883)

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

        alpha = 0.3

        client_indices = non_iid_dirichlet_partition(train_imgs, num_clientes, alpha=alpha)

        # Crear un subconjunto de datos para cada cliente en función de los índices
        client_subsets = [Subset(train_imgs, indices) for indices in client_indices] 
        plot_client_distributions(client_indices, train_imgs, "Dirichlet α=0.3")

    elif partition_type == "class_less": # Añdir Variable clases privadas

        alpha = 0.5

        clases_privadas = 0.4 # Porcentaje de clases privadas 

        # Reparto no-IID “class-less”:
        client_indices = non_iid_class_less_partition(train_imgs, num_clients=num_clientes, clases_privadas=clases_privadas, alpha=alpha)

        # Crear un subconjunto de datos para cada cliente a partir de los índices
        client_subsets = [Subset(train_imgs, indices) for indices in client_indices]

        # Visualizar la distribución (ningún cliente tendrá todas las clases y todos los clientes tendrán al menos una clase)
        plot_client_distributions(client_indices, train_imgs, f"Class-Less α={alpha}")

    else:
        print("Tipo de partición no soportado. Usa 'iid', 'dirichlet' o 'class_less'")
        exit()

    
    # Creamos los clientes
    for i in range(num_clientes):
        print(f"Cliente {i}: Inicializando...")

        if encrypted:
            ctx = ts.context_from(ctx_secret_key) # no contiene la clave secreta
        else:
            ctx = None
        clientes.append(Cliente(ROLANN(num_classes=10, encrypted=encrypted, context=ctx), client_subsets[i], device, client_id=i, broker="localhost", port=1883))

    for i, cliente in enumerate(clientes):
        print(f"Entrenando Cliente {i}...")
        cliente.training()  # Entrena localmente al cliente
        cliente.aggregate_parcial()  # Extrae las matrices locales del cliente

    time.sleep(5)  # Esperamos para asegurarnos de que los clientes han terminado de entrenar

    # Evalue un cliente random entre 0 y num_clientes-1
    idx = np.random.randint(0, num_clientes)

    print(f"Evaluando el modelo global con el cliente {idx}...")

    train_acc = clientes[idx].evaluate(train_loader)
    test_acc = clientes[idx].evaluate(test_loader)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Evaluamos el modelo global con otro cliente aleatorio diferente al anterior
    idx2 = np.random.randint(0, num_clientes)
    while idx2 == idx:
        idx2 = np.random.randint(0, num_clientes)

    print(f"Evaluando el modelo global con el cliente {idx2}...")

    train_acc2 = clientes[1].evaluate(train_loader)
    test_acc2 = clientes[1].evaluate(test_loader)

    print(f"Training Accuracy: {train_acc2:.4f}")
    print(f"Test Accuracy: {test_acc2:.4f}")

if __name__ == "__main__":
    main()
