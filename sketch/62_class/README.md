# 62-Class Sketch Diffusion Model

Modelo de difusión optimizado para 62 clases de sketches, 6x más grande que el modelo de 10 clases.

## 🚀 Uso Rápido

### 1. Preparar el entorno:
```bash
cd /home/ialbornoz/tesis/flow_and_diffusion/sketch/62_class
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sketch_diffusion
```

### 2. Iniciar entrenamiento:
```bash
./quick_start.sh /path/to/your/62/classes/dataset
```

## 📁 Estructura del Dataset

Tu dataset debe tener esta estructura:
```
/path/to/your/dataset/
├── class1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.png
│   └── ...
├── ...
└── class62/
    ├── image1.png
    └── ...
```

## ⚙️ Configuración Optimizada

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Resolución** | 64x64 | Alta resolución para mejor calidad |
| **Canales** | [128, 256, 512, 1024] | 6x más grande que modelo de 10 clases |
| **Capas Residuales** | 6 | Mayor capacidad para 62 clases |
| **Batch Size** | 64 | Como solicitado |
| **Learning Rate** | 3e-5 | Optimizado para estabilidad |
| **Épocas** | 20,000 | Entrenamiento extendido |

## 🔧 Monitoreo

### Conectar a la sesión tmux:
```bash
tmux attach -t sketch_62class_EXPERIMENT_ID
```

### Ver logs de entrenamiento:
```bash
tail -f experiments/EXPERIMENT_ID/logs/training_sketch_62classes.log
```

### Ver logs de monitoreo del sistema:
```bash
tail -f experiments/EXPERIMENT_ID/monitoring/monitor.log
```

### Ver métricas del sistema:
```bash
cat experiments/EXPERIMENT_ID/monitoring/system_metrics.json
```

## 📊 Monitoreo Automático

El sistema registra automáticamente:
- **CPU**: Uso de procesador (%)
- **Memoria**: RAM utilizada (GB y %)
- **GPU**: Memoria GPU (GB y %)
- **Disk I/O**: Lectura/escritura (MB/s)
- **Network I/O**: Tráfico de red (MB/s)

**Archivos generados:**
- `experiments/EXPERIMENT_ID/monitoring/system_metrics.json` - Métricas en formato JSON
- `experiments/EXPERIMENT_ID/monitoring/system_metrics_plot.png` - Gráficos de métricas
- `experiments/EXPERIMENT_ID/monitoring/monitor.log` - Log detallado de monitoreo

## 🔧 Comandos Útiles

```bash
# Listar sesiones tmux
tmux list-sessions

# Ver logs en tiempo real
tail -f experiments/EXPERIMENT_ID/logs/training_sketch_62classes.log

# Terminar sesión
tmux kill-session -t sketch_62class_EXPERIMENT_ID
```

## 📁 Estructura de Archivos del Experimento

```
experiments/EXPERIMENT_ID/
├── ckps/                    # Checkpoints del modelo
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_epoch_XXXX.pth
├── samples/                 # Muestras generadas
│   ├── samples_epoch_XXXX_classes_0-61.png
│   └── class_mapping_epoch_XXXX.txt
├── logs/                    # Logs de entrenamiento
│   ├── training_sketch_62classes.log
│   └── losses_sketch_62classes.npy
├── monitoring/              # Métricas del sistema
│   ├── system_metrics.json
│   ├── system_metrics_plot.png
│   └── monitor.log
└── config.json             # Configuración del experimento
```

## 🎯 Resumen

- **Una sola configuración optimizada** para 62 clases
- **Modelo 6x más grande** que el de 10 clases
- **Batch size 64** como solicitado
- **Monitoreo automático** de métricas del sistema
- **Entrenamiento con tmux** para persistencia
- **Fácil de usar** con un solo comando