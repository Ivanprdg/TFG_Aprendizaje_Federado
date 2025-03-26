# 🧠 TFG - Aprendizaje Federado

📅 **Actualización:** 25/03  
🔧 **Rama:** `feature/Implementar_vector_caracteristicas`

---

## 📊 Datasets

Se han incorporado los datasets **MNIST** y **CIFAR10**. Para elegir cuál utilizar, se ha definido una variable `dataset` que puede tomar los valores:
- `0` → MNIST
- `1` → CIFAR10

La función `get_dataset(dataset: int)` se encarga de:
- Cargar el dataset correspondiente
- Adaptar las imágenes al formato requerido por ResNet18
- Calcular y aplicar la normalización de forma automática

---

## 🔍 Implementación de ResNet18 + ROLANN

Se utiliza una **ResNet18 preentrenada** como extractor de características.  
La capa `fc` final ha sido reemplazada por una capa identidad (`nn.Identity()`), lo que permite obtener directamente el **vector de características** de la imagen.

La clasificación se realiza con el modelo **ROLANN**, que aprende a partir de dichos vectores extraídos.

---

### 🗂️ Adaptación y Normalización de Datos

#### ✅ Preprocesamiento automático
- Se ha calculado la **media** y la **desviación típica** de los valores de píxel dividiendo primero las imágenes por 255 (para llevar sus valores al rango [0, 1]). Luego se calcula el promedio y la desviación de todos los píxeles del dataset, incluyendo todas las imágenes, la altura y la anchura **axis=(0,1,2)**:

  ```python
  mean = (dataset.data / 255).mean(axis=(0, 1, 2))
  std = (dataset.data / 255).std(axis=(0, 1, 2))
  ```

  Esto permite obtener una referencia común de brillo y contraste que se usará después para normalizar las imágenes. Así, los datos estarán centrados y distribuidos de forma estable para que la red los entienda mejor.

- En el caso de **MNIST**, al tener un único canal, se replica el valor 3 veces para simular una imagen RGB compatible con ResNet18:

  ```python
  mean_rgb = [mean] * 3
  std_rgb = [std] * 3
  ```

- Estos valores se utilizan posteriormente en la transformación de normalización con:

  ```python
  transforms.Normalize(mean_rgb, std_rgb)
  ```

#### 🖼️ MNIST
- Imágenes originalmente en B/N (1 canal, 28x28)
- Adaptadas para ResNet18:
  - Redimensionadas a **224x224**
    ```python
    transforms.Resize((224, 224))
    ```
  - Convertidas a **RGB** para que tengan 3 canales:
    ```python
    transforms.Grayscale(num_output_channels=3)
    ```

#### 🖼️ CIFAR10
- Imágenes ya en RGB, solo se redimensionan:
  ```python
  transforms.Resize((224, 224))
  ```

---

### ⚙️ Congelación y Modo Evaluación de ResNet18

- La ResNet se congela para que no se actualicen sus pesos durante el entrenamiento:
  ```python
  for param in resnet.parameters():
      param.requires_grad = False
  ```

- Se activa el modo evaluación para que la ResNet deje de comportarse como si estuviera entrenando. Esto hace que capas como BatchNorm no cambien
  su funcionamiento y mantengan los resultados estables:
  ```python
  resnet.eval()
  ```

  ResNet18 contiene capas llamadas **Batch Normalization**, que ajustan internamente los datos con cada pasada.  
  Aunque congelemos los pesos del modelo, estas capas **siguen actualizando estadísticas internas** (como la media y desviación) cada vez que se le pasan imágenes, por ejemplo en:
  ```python
  for x, y in train_loader:
      caracteristicas = resnet(x)
  ```
  Esto puede provocar que las características extraídas cambien ligeramente con cada iteración, afectando al modelo ROLANN.

  Usar `resnet.eval()` **frena esa actualización automática**, asegurando que la salida de ResNet sea siempre la misma durante el entrenamiento.

- Durante la extracción de características, se desactiva el cálculo de gradientes:
  ```python
  with torch.no_grad():
      # Extracción de características
  ```

---

## ⚡ Uso de GPU

Se utiliza GPU si está disponible, tanto para ResNet como para ROLANN:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## 📈 Resultados

### **22/03** (antes de normalizar correctamente):

- **MNIST**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.9475  |
| Test Accuracy      | 0.9492  |

- **CIFAR10**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.7740  |
| Test Accuracy      | 0.7703  |

---

### **25/03** (con normalización automática y mejoras):

- **MNIST**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.9678  |
| Test Accuracy      | 0.9664  |

- **CIFAR10**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.8460  |
| Test Accuracy      | 0.8372  |

---

## ✅ Conclusiones

- La incorporación de la **normalización automática** basada en el propio dataset mejora significativamente el rendimiento, especialmente en el caso de CIFAR10.

## 📖 Bibliografía y Fuentes

- [Eliminar capa de clasificacion en RESNET](https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch)
- [Transformaciones en datasets](https://pytorch.org/vision/0.9/transforms.html)