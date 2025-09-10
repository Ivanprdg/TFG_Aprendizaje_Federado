# 🧠 TFG - Aprendizaje Federado

📅 **Actualización:** 10/09/2025  
🔧 **Rama:** `Implementar_version_federada_local`

---

## 📊 Datasets

El proyecto soporta los datasets **MNIST**, **CIFAR10** y **CIFAR100**.  
La función `get_dataset(dataset: int)` permite seleccionar el dataset y devuelve los objetos necesarios para el entrenamiento y evaluación.

- `0` → MNIST
- `1` → CIFAR10
- `2` → CIFAR100

El preprocesamiento incluye:
- Redimensionado a 224x224 píxeles.
- Conversión a RGB si es necesario.
- Normalización automática calculada sobre el propio dataset.

---

## 🔍 Arquitectura del Proyecto

El sistema implementa **aprendizaje federado** usando PyTorch, ResNet18 y el modelo ROLANN.  
Cada cliente tiene su propia instancia de ResNet18 (congelada y en modo evaluación) y una instancia de ROLANN.

- **ResNet18** se usa como extractor de características.
- **ROLANN** realiza la clasificación sobre los vectores extraídos.

---

## 🚀 Implementación Federada

### División del dataset

El dataset se divide entre los clientes usando diferentes estrategias:
- **IID**: División equitativa y aleatoria.
- **Dirichlet**: División no-IID controlando la heterogeneidad con el parámetro α.
- **Class-less**: Cada cliente tiene acceso solo a un subconjunto de clases.

### Cliente

Cada cliente:
- Entrena localmente su modelo ROLANN usando las características extraídas por su ResNet.
- Publica sus matrices locales (`M`, `U`, `S`) al coordinador mediante MQTT.

### Coordinador

El coordinador:
- Recibe las actualizaciones de todos los clientes.
- Realiza la agregación global usando SVD sobre las matrices recibidas.
- Actualiza el modelo global y publica el resultado para que los clientes lo descarguen.

---

## ⚡ Uso de GPU

El sistema detecta automáticamente si hay GPU disponible y mueve los modelos y datos al dispositivo adecuado:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## 📡 Comunicación MQTT

La comunicación entre clientes y coordinador se realiza mediante **MQTT** (broker Mosquitto):

- Cada cliente publica en:  
  ```
  federated/client/<client_id>/update
  ```
- El coordinador se suscribe a:  
  ```
  federated/client/+/update
  ```
- El modelo global se publica en:  
  ```
  federated/global_model
  ```

Los datos se serializan con **pickle**, se codifican en **base64** y se envuelven en **JSON** para su transmisión.

---

## 🗂️ Estructura de Carpetas

Ejemplo de estructura de archivos del proyecto:

```
Codigo_tfg/
├── Cliente.py
├── Coordinador.py
├── ROLANN.py
├── fedHEONN.py
├── datasets.py
├── main.py
├── README.md
└── data/
```

---

## ✅ Notas

- El proyecto está preparado para experimentar con diferentes particiones de datos y cifrado homomórfico (TenSEAL).
- La sincronización entre clientes y coordinador se realiza mediante barreras de threading.
- El código está modularizado para facilitar la extensión y pruebas.