import torch
from ROLANN import ROLANN
from Cliente import Cliente
from Coordinador import Coordinador
from datasets import get_dataset
from torch.utils.data import random_split

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
    num_clientes = 4
    clientes = [] # Lista de clientes

    # Cada cliente tendrá un subconjunto del dataset, su propia ResNet y su propio ROLANN
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

    # División del dataset en partes iguales para cada cliente
    dataset_dividido = random_split(train_imgs, batch_size)
    
    # Creamos los clientes
    for i in range(num_clientes):
        print("Cliente " + str(i) + ": Inicializando el cliente...")
        clientes.append(Cliente(ROLANN(num_classes=10), dataset_dividido[i], device))


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
