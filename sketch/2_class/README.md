# 2-Class Sketch Diffusion Experiments

Esta carpeta contiene todos los archivos necesarios para entrenar modelos de difusión de sketches con solo 2 clases.

## Archivos incluidos

- `train_experiment.py` - Script principal de entrenamiento
- `experiment_manager.py` - Gestor de experimentos con configuraciones para 2 clases
- `sketchutils.py` - Utilidades modificadas para 2 clases (null class = 2)
- `sketch_model.py` - Modelo de datos modificado para usar solo 2 clases
- `sketch_unet.py` - Arquitectura UNet modificada para 3 clases total (2 + 1 null)
- `start_2class_experiment.sh` - Script de inicio para experimentos de 2 clases

## Configuraciones disponibles

### sketch_32x32_2classes
- **Descripción**: Modelo de investigación con 2 clases
- **Resolución**: 32x32 píxeles
- **Arquitectura**: [64, 128, 256, 512] canales, 4 capas residuales
- **Parámetros**: 
  - Learning rate: 1e-4
  - Batch size: 32
  - Epochs: 10000
  - Checkpoint interval: 50
  - Sample interval: 50

### sketch_32x32_2classes_simple
- **Descripción**: Modelo simplificado para entrenamiento más rápido
- **Resolución**: 32x32 píxeles
- **Arquitectura**: [32, 64, 128, 256] canales, 2 capas residuales
- **Parámetros**:
  - Learning rate: 1e-4
  - Batch size: 64
  - Epochs: 5000
  - Checkpoint interval: 25
  - Sample interval: 25

## Uso

### Iniciar un experimento
```bash
./start_2class_experiment.sh sketch_32x32_2classes
```

### Monitorear el entrenamiento
```bash
screen -r sketch_2class_<experiment_id>
```

### Ver logs en tiempo real
```bash
tail -f experiments/<experiment_id>/logs/training_<experiment_name>_*.log
```

## Diferencias con el experimento de 50 clases

1. **Número de clases**: Solo 2 clases en lugar de 50
2. **Null class**: Usa clase 2 como null class en lugar de clase 50
3. **Embedding**: El modelo UNet usa solo 3 embeddings (2 clases + 1 null)
4. **Configuraciones**: Optimizadas para entrenamiento con menos clases

## Estructura de experimentos

Los experimentos se guardan en:
```
experiments/
└── <experiment_name>_<timestamp>/
    ├── ckps/          # Checkpoints del modelo
    ├── samples/       # Muestras generadas durante el entrenamiento
    ├── logs/          # Logs de entrenamiento
    ├── results/       # Resultados finales
    └── config.json    # Configuración del experimento
```
