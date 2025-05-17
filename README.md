# 🧠 TFG - Aprendizaje Federado

📅 **Actualización:** 07/04/2024  
🔧 **Rama:** `Implementar_version_federada_local`

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

## 🚀 **Implementación Federada**

### 🔄 División del dataset entre clientes
El dataset se divide equitativamente entre múltiples clientes:
```python
dataset_dividido = random_split(dataset, batch_size)
```

### 📡 Cliente
Cada cliente posee:
- Una instancia propia de ResNet18 (congelada y en modo evaluación).
- Una instancia propia del modelo ROLANN.

Cada cliente entrena localmente su instancia de ROLANN usando las características extraídas por su ResNet local y posteriormente envía sus matrices locales (`M`, `U`, `S`) al Coordinador.

### 🗃️ Coordinador
- Recibe y acumula matrices (`M`, `U`, `S`) de cada cliente.
- Realiza la agregación global usando SVD sobre las matrices recibidas.
- Actualiza la instancia global del modelo ROLANN con las matrices agregadas.

### 🛠️ Manejo de dispositivos
- El coordinador asegura que todas las matrices se encuentren en el mismo dispositivo (CPU/GPU) antes de realizar cálculos, evitando errores de incompatibilidad.
```python
mg_tensor_list = [m if isinstance(m, torch.Tensor) else torch.tensor(m, device=self.device) for m in mg_list]
```


## 🚀 Comunicación

Para la orquestación cliente–coordinador usamos **MQTT**:

1. **Broker**  
   - Actualmente usamos **Mosquitto** como broker MQTT, ejecutándose en `localhost:1883`
2. **Topics**  
   - Cada cliente publica sus actualizaciones en:  
     ```
     federated/client/<client_id>/update
     ```
   - El coordinador se suscribe a:
     ```
     federated/client/+/update
     ```
     donde `+` es un **wildcard** que coincide con cualquier `client_id` en ese nivel. Así recibe todas las actualizaciones sin tener que suscribirse manualmente a cada cliente. Los clientes se suscriben a `federated/global_model`.
3. **Wildcards**  
   - `+` sustituye a un solo nivel jerárquico (p.ej., `client/0/update`, `client/1/update`).  
   - `#` sustituye a todos los niveles siguientes (p.ej., `sensors/#` recibiría `sensors/temperature`, `sensors/humidity/room1`, etc.).
4. **QOS (Quality of Service)**  
   - Usamos `qos=1` para asegurar al menos una entrega (puede reenviar en caso de fallo).
5. **Payload**  
   - Serializamos los objetos (vectores y matrices) con **pickle**, luego los codificamos a **base64** y envolvemos en **JSON**.  
   - Ejemplo en el cliente, para enviar `M_enc` (puede ser CKKSVector o Tensor) y `US` (Tensor):
     ```python
     body = []
     for M_enc, US in zip(local_M, local_US):
         # Si está cifrado, se queda en CKKSVector, sino tensor
         m_obj = M_enc.decrypt() if hasattr(M_enc, "decrypt") else M_enc.cpu().numpy().tolist()
         us_obj = US.cpu().numpy().tolist()
         bM  = base64.b64encode(pickle.dumps(m_obj)).decode()
         bUS = base64.b64encode(pickle.dumps(us_obj)).decode()
         body.append({"M": bM, "US": bUS})
     mqtt.publish(f"federated/client/{self.client_id}/update", json.dumps(body), qos=1)
     ```
   - En el coordinador, al recibir:
     ```python
     def _on_client_update(self, client, userdata, msg):
         data = json.loads(msg.payload.decode())
         M_list, US_list = [], []
         for entry in updates:
             M_obj = pickle.loads(base64.b64decode(entry["M"]))
             US_obj = pickle.loads(base64.b64decode(entry["US"]))
             M_list.append(torch.tensor(M_obj, device=self.device))
             US_list.append(torch.tensor(US_obj, device=self.device))
         self.recolect_parcial(M_list, US_list)
     ```
   - **¿Por qué pickle/base64/JSON?**  
     - `pickle` serializa objetos Python arbitrarios (incluidos CKKSVector).  
     - `base64` convierte los bytes binarios a texto para poder incluirlos en un mensaje JSON.  
     - `JSON` es legible y ampliamente soportado por MQTT.
6. **Flujo de datos**  
   1. **Cliente** entrena localmente, calcula `M` y `US`, serializa y hace `publish`.  
   2. **Broker** reenvía al **Coordinador**, que tiene un callback `_on_client_update`.  
   3. **Coordinador** deserializa, agrega con SVD, recalcula pesos con `_calculate_weights()` y finalmente sirve el modelo global para evaluación.
   4. **Esquema:**
   ```
    +----------------+         publish                 +-----------+        on_message         +----------------+
    |   Cliente 0    | ──────────────────────────────▶ |           | ────────────────────────▶|               │
    | (train + agg)  |   federated/client/0/update     |           |   _on_client_update     |               │
    +----------------+                                 |  Broker   |                         | Coordinador   |
                                                      |           |                         |               │
    +----------------+         publish                 |           |        on_message       |               │
    |   Cliente 1    | ──────────────────────────────▶ |           | ────────────────────────▶|               │
    | (train + agg)  |   federated/client/1/update     |           |                         |               │
    +----------------+                                 +-----------+                         +----------------+
          ⋮                                                                                      ▲
          ⋮                                                                                      │ after all updates
          ⋮                                                                                      │ call recolect_parcial()
    +----------------+                                                                             │
    |   Cliente N    |         publish                                                                    
    | (train + agg)  | ──────────────────────────────▶ Broker                                          
    +----------------+   federated/client/N/update                                                      
                                                                                                    │
                                                                                                    │ publish
                                                                                                    │ federated/global_model
                                                                                                    ▼
    +----------------+                                 +-----------+                         +----------------+
    |   Cliente 0    | ◀───────────────────────────── |           | ◀─────────────────────── | Coordinador   |
    |  _on_global_   |      federated/global_model    |           |    publish global       |  (after agg)  |
    +----------------+                                 |           |                         +----------------+
                                                      |           |
    +----------------+                                 |           |
    |   Cliente 1    | ◀───────────────────────────── |           |
    |  _on_global_   |      federated/global_model    |           |
    +----------------+                                 +-----------+
          ⋮
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

### Versión Federada (07/04):

- **MNIST**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.9685  |
| Test Accuracy      | 0.9673  |

- **CIFAR10**:

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.8459  |
| Test Accuracy      | 0.8384  |


### Resultados de la Experimentación

#### Dataset: MNIST

|   Número de Clientes | Tipo de Partición   |   Training Accuracy |   Test Accuracy |
|----------------------|---------------------|---------------------|-----------------|
|                    4 | Dirichlet (α=0.3)   |              0.9684 |          0.9664 |
|                    4 | Class Less (α=0.5)  |              0.9685 |          0.9685 |
|                    8 | Dirichlet (α=0.3)   |              0.9687 |          0.967  |
|                    8 | Class Less (α=0.5)  |              0.9686 |          0.9671 |
|                   12 | Dirichlet (α=0.3)   |              0.9686 |          0.967  |
|                   12 | Class Less (α=0.5)  |              0.9686 |          0.9670 |

#### Dataset: CIFAR-10

|   Número de Clientes | Tipo de Partición   |   Training Accuracy |   Test Accuracy |
|----------------------|---------------------|---------------------|-----------------|
|                    4 | Dirichlet (α=0.3)   |              0.8456 |          0.8376 |
|                    4 | Class Less (α=0.5)  |              0.8456 |          0.8381 |
|                    8 | Dirichlet (α=0.3)   |              0.8455 |          0.8379 |
|                    8 | Class Less (α=0.5)  |              0.8458 |          0.8373 |
|                   12 | Dirichlet (α=0.3)   |              0.8457 |          0.838  |
|                   12 | Class Less (α=0.5)  |              0.8456 |          0.8381 |


---

## ✅ Conclusiones

- La incorporación de la **normalización automática** basada en el propio dataset mejora significativamente el rendimiento, especialmente en el caso de CIFAR10.
- La implementación federada muestra un rendimiento equivalente o ligeramente mejor comparado con el enfoque centralizado, validando así su eficacia.

## 📖 Bibliografía y Fuentes

- [Eliminar capa de clasificacion en RESNET](https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch)
- [Transformaciones en datasets](https://pytorch.org/vision/0.9/transforms.html)
- [Uso de random_split](https://pytorch.org/docs/stable/data.html)