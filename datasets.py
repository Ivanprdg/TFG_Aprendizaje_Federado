from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_dataset(dataset: int):

    if dataset == 0:

        # MNIST
        # Transformación temporal para calcular la media y la desviación típica
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        # Cargamos el dataset MNIST con la transformación temporal
        mnist_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

        # Calcular la media y la desviación típica
        mnist_mean = (mnist_train_dataset.data / 255).mean(axis=(0, 1, 2))
        mnist_std = (mnist_train_dataset.data / 255).std(axis=(0, 1, 2))

        # Convertir a listas con 3 canales
        mnist_mean_rgb = [mnist_mean] * 3
        mnist_std_rgb = [mnist_std] * 3

        print("MNIST mean RGB:", mnist_mean_rgb)
        print("MNIST std RGB:", mnist_std_rgb)

        # Transformación definitiva
        transform.transforms.append(
            transforms.Normalize(mean=mnist_mean_rgb, std=mnist_std_rgb)
        )

        train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        test_loader = DataLoader(test, batch_size=128, shuffle=False)

    elif dataset == 1:

        # CIFAR10
        # Transformación temporal para calcular la media y la desviación típica
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Cargamos el dataset CIFAR10 con la transformación temporal
        cifar_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

        cifar_mean = (cifar_train_dataset.data / 255).mean(axis=(0, 1, 2))
        cifar_std = (cifar_train_dataset.data / 255).std(axis=(0, 1, 2))

        print("CIFAR10 mean RGB:", cifar_mean)
        print("CIFAR10 std RGB:", cifar_std)

        # Transformación definitiva
        transform.transforms.append(
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        )
 
        train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        test_loader = DataLoader(test, batch_size=128, shuffle=False)

    elif dataset == 2:

        # CIFAR100
        # Transformación temporal para calcular la media y la desviación típica
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Cargamos el dataset CIFAR100 con la transformación temporal
        cifar100_train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)

        cifar100_mean = (cifar100_train_dataset.data / 255).mean(axis=(0, 1, 2))
        cifar100_std = (cifar100_train_dataset.data / 255).std(axis=(0, 1, 2))

        print("CIFAR100 mean RGB:", cifar100_mean)
        print("CIFAR100 std RGB:", cifar100_std)

        # Transformación definitiva
        transform.transforms.append(
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        )

        train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        test_loader = DataLoader(test, batch_size=128, shuffle=False)
    else:
        raise ValueError("Valor inválido para el dataset. Usa 0 (MNIST) o 1 (CIFAR10) o 2 (CIFAR100).")

    return train, train_loader, test_loader