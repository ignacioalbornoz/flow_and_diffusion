# Modelo 256x256 - Cambios y Configuración

## 📋 Cambios Realizados para 256x256

### **1. Nueva Configuración de Experimento (`experiment_manager.py`)**

Agregué la configuración `sketch_256x256_2classes`:

```python
"sketch_256x256_2classes": {
    "description": "256x256 resolution, 2 classes, research-grade model (original dataset size)",
    "image_size": 256,                    # ← Cambio: 32 → 256
    "model_channels": [128, 256, 512, 1024],  # ← Cambio: [64, 128, 256, 512] → [128, 256, 512, 1024]
    "num_residual_layers": 6,             # ← Cambio: 4 → 6
    "learning_rate": 5e-5,                # ← Cambio: 1e-4 → 5e-5
    "guidance_scale": 1.5,                # ← Sin cambio
    "eta": 0.1,                          # ← Sin cambio
    "weight_multiplier": 1.0,             # ← Sin cambio
    "batch_size": 8,                      # ← Cambio: 32 → 8
    "num_epochs": 10000,                  # ← Sin cambio
    "checkpoint_interval": 0,             # ← Cambio: 50 → 0 (solo best model)
    "sample_interval": 25                 # ← Cambio: 50 → 25 (más frecuente)
}
```

### **2. Actualización del Dataset (`sketch_model.py`)**

```python
def get_image_size(self):
    """Get the current image size"""
    return 256  # ← Cambio: 32 → 256 (tamaño original del dataset)
```

Y actualicé el comentario:
```python
samples = torch.cat(samples, dim=0)  # (num_samples, 1, 256, 256)  # ← Cambio: 32, 32 → 256, 256
```

### **3. Cambio de Directorio del Dataset**

```python
def __init__(self, data_dir: str = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch", max_classes: int = 2):
    # ← Cambio: "/home/shared_data/Datasets/sketchy/train/sketch" → directorio local
```

### **4. Dataset Combinado (Nuevo)**

- **Antes**: 1,148 imágenes (638 airplane + 510 alarm_clock)
- **Después**: 7,150 imágenes (3,639 airplane + 3,511 alarm_clock)
- **Fuentes**: Sketchy original + Quickdraw Extended

## 🔍 **Razones de los Cambios**

### **Arquitectura del Modelo:**
- **Canales**: `[64, 128, 256, 512]` → `[128, 256, 512, 1024]`
  - **Razón**: 256x256 requiere más capacidad para capturar detalles finos
- **Capas residuales**: `4` → `6`
  - **Razón**: Mayor profundidad para procesar imágenes de alta resolución

### **Parámetros de Entrenamiento:**
- **Learning rate**: `1e-4` → `5e-5`
  - **Razón**: Entrenamiento más estable en alta resolución
- **Batch size**: `32` → `8`
  - **Razón**: Limitaciones de memoria GPU (256x256 usa ~64x más memoria)
- **Checkpoint interval**: `50` → `0`
  - **Razón**: Solo guardar best model, no checkpoints intermedios (ahorro de espacio)
- **Sample interval**: `50` → `25`
  - **Razón**: Monitoreo más frecuente del progreso de entrenamiento

### **Dataset:**
- **Tamaño**: 32x32 → 256x256 (tamaño original)
- **Cantidad**: 1,148 → 7,150 imágenes (6x más datos)
- **Fuentes**: Solo Sketchy → Sketchy + Quickdraw

## 📊 **Impacto en el Modelo**

| Aspecto | 32x32 | 256x256 | Cambio |
|---------|-------|---------|--------|
| **Parámetros** | 34,947,534 | 201,580,462 | +477% |
| **Memoria GPU** | ~1GB | ~24GB | +2400% |
| **Tiempo por época** | ~1s | ~30-60s | +3000-6000% |
| **Calidad esperada** | Básica | Alta resolución | Significativa |

## 🚀 **Uso**

### **Entrenar el modelo:**
```bash
conda activate sketch_diffusion
python train_experiment.py sketch_256x256_2classes
```

### **Generar muestras:**
```bash
python generate_samples.py sketch_256x256_2classes_YYYYMMDD_HHMMSS
```

## 📁 **Estructura del Dataset Combinado**

```
/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch/
├── airplane/
│   ├── n02691156_XXXX-X.png          # Imágenes originales Sketchy
│   └── quickdraw_XXXXXXXXXXXX.png    # Imágenes Quickdraw Extended
└── alarm_clock/
    ├── n02691156_XXXX-X.png          # Imágenes originales Sketchy
    └── quickdraw_XXXXXXXXXXXX.png    # Imágenes Quickdraw Extended
```

## ⚠️ **Consideraciones Importantes**

1. **Memoria GPU**: Requiere ~24GB de VRAM
2. **Tiempo de entrenamiento**: Significativamente más largo que 32x32
3. **Almacenamiento**: Modelo de ~2.4GB por checkpoint
4. **Calidad**: Mucho mejor detalle y resolución que 32x32

## 🔧 **Configuración Optimizada**

- **Solo best model**: No se guardan checkpoints intermedios
- **Muestras frecuentes**: Generación cada 25 épocas para monitoreo
- **Early stopping**: Parada temprana si no mejora en 2500 épocas
- **Batch size reducido**: Optimizado para memoria disponible
