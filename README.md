# 🧠 TFG - Aprendizaje Federado

📅 **Actualización:** 22/03  
🔧 **Rama:** `feature/Implementar_vector_caracteristicas`

---
## 📊 Datasets

Se ha añadido el dataset de CIFAR10. Para poder elegir el dataset a utilizar se ha designado una variable **dataset** la cual puedes tomar valores
0 (MNIST) o 1 (CIFAR).

## 🔍 Implementación de RESNET

Se ha utilizado una **ResNet18 preentrenada** para aprovechar su primera capa de extracción de características.  
La última capa de clasificación ha sido eliminada, ya que **ROLANN** se encarga de la clasificación. Esta capa se ha sustituido por una **capa de identidad**, permitiendo que el vector de salida sea igual al de entrada: el **vector de características**.

### 🗂️ Adaptación de datos
- Las imágenes de **MNIST** (originalmente en B/N y de tamaño 28x28) fueron adaptadas para ser compatibles con ResNet18:
  - Redimensionadas a **224x224**
  - Convertidas a formato **RGB** mediante:
    ```python
    transforms.Grayscale(num_output_channels=3)
    ```

### ⚖️ Congelación de pesos
- Al ser un modelo preentrenado, **no queremos actualizar sus pesos**, por lo que se congelan con:
  ```python
  for param in resnet.parameters():
      param.requires_grad = False
  ```

- Además, para evitar el cálculo del gradiente durante el entrenamiento:
  ```python
  with torch.no_grad():
      # Extracción de características
  ```

### ⚡ Uso de GPU
- Para mejorar el rendimiento, se utiliza **GPU** moviendo las operaciones pesadas mediante:
  ```python
  model.to(device)
  ```

---

## 📈 Resultados Iniciales

| Métrica            | Valor   |
|--------------------|---------|
| Training Accuracy  | 0.9475  |
| Test Accuracy      | 0.9492  |
