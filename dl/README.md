# Rama de Deep Learning

Clasificadores de huevos sobre los mismos recortes y el mismo split que la rama ML: CNN base,
VGG16, red liviana (EfficientNet-B0 / MobileNetV3) y Vision Transformer, con transfer learning y
Grad-CAM. También el detector YOLO11n de 3 clases (pedido por el profesor de DL como baseline:
una sola red que detecta y clasifica la especie a la vez, sin un clasificador aparte) y, para
comparar, el mismo detector de 1 clase. Entregables y checklist de la materia:
[`docs/entregas/deep-learning/`](../docs/entregas/deep-learning/README.md).

## Notebooks de YOLO

| Notebook | Qué compara | Cómo correrlo |
|---|---|---|
| `notebooks/yolo_1class.ipynb` | Detector de 1 clase (`item` = huevo), sin aumentos vs. con aumentos | `sbatch scripts/run_yolo_1class.sh` |
| `notebooks/yolo_3class.ipynb` | Detector de 3 clases (ascaris/trichuris/uncinaria), sin aumentos vs. con aumentos, más qué tan bien clasifica la especie | `sbatch scripts/run_yolo_3class.sh` |

Los dos entrenan en GPU (`accel`), semilla 42, y ejecutan el notebook completo con
`jupyter nbconvert --execute` (igual que `detector/scripts/run_compare_detectors.sh`). Las corridas
quedan en `dl/outputs/yolo/<nombre>/`; si los pesos ya existen, el notebook no reentrena.

Las rutas de `data/raw/data.yaml` y `data/aug/data.yaml` ya apuntan a este repo
(`/home/scar-test/deep-learning/...`).

## Notebooks de clasificación

| Notebook | Qué hace | Cómo correrlo |
|---|---|---|
| `notebooks/cnn_base.ipynb`, `notebooks/cnn_combos.ipynb` | Búsqueda de la CNN desde cero (rondas 1 y 2) | `scripts/run_cnn_base.sh`, `scripts/run_cnn_combos.sh` |
| `notebooks/cnn_final.ipynb` | CNN base final (5 bloques, lr 3e-4), 100 épocas, 3 semillas | `sbatch scripts/run_cnn_final.sh` |
| `notebooks/transfer_learning.ipynb` | VGG16 preentrenada en ImageNet (o EfficientNet-B0 con `ARCH=efficientnet_b0`), ajuste en dos fases (PLAN §6.3), 3 semillas, comparada con la CNN base | `sbatch scripts/run_transfer.sh` |

**Antes del primer `run_transfer.sh`**, en el nodo de login (los nodos de cómputo no tienen
internet), bajar los pesos de ImageNet:
`python -c "from torchvision import models; models.vgg16(weights='IMAGENET1K_V1'); models.efficientnet_b0(weights='IMAGENET1K_V1')"`.
Las corridas quedan en `dl/outputs/transfer_<arch>/seed<N>/`.

VGG16 ejecuta `notebooks/transfer_learning.ipynb`. Con otra `ARCH`, el script ejecuta una copia
nueva, `notebooks/transfer_learning_<arch>.ipynb` (se regenera desde el de VGG16 en cada corrida),
así las dos redes pueden entrenar a la vez sin pisarse:

```bash
sbatch dl/scripts/run_transfer.sh                                        # VGG16, GPU
sbatch --export=ALL,ARCH=efficientnet_b0 dl/scripts/run_transfer.sh      # EfficientNet-B0, GPU
sbatch --export=ALL,ARCH=efficientnet_b0 dl/scripts/run_transfer_cpu.sh  # EfficientNet-B0, CPU (bigmem)
```

Los cambios del notebook se hacen en el de VGG16; la copia de EfficientNet solo guarda su corrida.

```
dl/
├── src/         código de los modelos, el entrenamiento y la evaluación
├── notebooks/   exploración y resultados
├── scripts/     trabajos de Slurm (accel)
└── outputs/     (no versionado) pesos, curvas y corridas
```

## Lo que consume de las otras carpetas

| Qué | De dónde |
|---|---|
| Recortes (ImageFolder + `crops.csv`) y split | `data/crops/` |
| Recortes de lo que detecta YOLO | `detector/outputs/detected/` |

## Reglas

- Mismo split que todo el proyecto; `test` congelado hasta la evaluación final.
- Un experimento por corrida, con su configuración y semilla guardadas en `outputs/`.
- Guardar pesos, configuración y partición de cada experimento (receta del PLAN §6.3).
