# Geohelmintos en microscopía — detección, clasificación y conteo

Sistema de apoyo al coprológico para comunidades del Chocó: a partir de fotos de
microscopio tomadas con celular, detecta, clasifica y cuenta huevos de *Ascaris
lumbricoides*, *Trichuris trichiura* y uncinarias, sin necesidad de internet.

Proyecto final de **Machine Learning** y **Redes Neuronales y Deep Learning** (EAFIT
2026-2), en alianza con Microbiología y Parasitología de la UPB. Equipo: Samuel Llano
Madrigal, Laura Andrea Castrillon y Jerónimo Campuzano.

**Contexto completo, arquitectura y decisiones: [docs/PLAN.md](docs/PLAN.md).**

```
foto → YOLO (detecta huevos) → recortes ─┬─ rama ML: DINOv2 + ML clásico
                                         └─ rama DL: CNN / VGG16 / red liviana / ViT
                                → conteo por lámina (HPG) → app offline
```

## Estructura

Una carpeta por pieza del sistema, siguiendo el recorrido del proyecto. Cada una lleva su código
(`src/`), sus notebooks, sus scripts de Slurm (`scripts/`) y lo que genera (`outputs/`, no
versionado), y tiene su propio README.

```
├── docs/         PLAN.md (plan maestro) y entregas/ por materia
├── data/         (no versionado) los datos: roboflow, raw, aug, crops, external; ver data/README.md
├── logs/         logs de Slurm por componente: logs/{data_prep,detector,ml,dl}/ (los .out no se versionan)
│
├── data_prep/    de Chula a recortes: descarga, filtro, recorte, aumentos, EDA de las fotos
├── detector/     YOLO: entrenamiento, recortes de lo que detecta, comparación de modelos
├── ml/           rama ML: embeddings DINOv2, EDA de embeddings, modelado
├── dl/           rama DL: CNN, VGG16, EfficientNet, ViT, Grad-CAM
└── app/          app offline (ONNX + interfaz local)
```

```
foto → data_prep → data/ ─┬─ detector (YOLO) → recortes ─┬─ ml (DINOv2 + ML clásico)
                          └──────────────────────────────┴─ dl (CNN / VGG16 / red liviana / ViT)
                                                          → app (conteo por lámina)
```

## Entorno

- **Mac (desarrollo):** conda `machine-learning-test` (Python 3.14, torch con MPS).
- **Apolo III (entrenamiento):** partición `accel` (2× H100). Los nodos de cómputo no
  tienen internet: descargar datos y pesos desde el nodo de login. Los `sbatch` se lanzan
  desde la raíz del repo y cada uno escribe su log en `logs/<componente>/<trabajo>-<id>.out`
  (`#SBATCH --output`), así que no quedan `slurm-*.out` sueltos en la raíz.

```bash
pip install -r requirements.txt
```

## Pipeline de datos

Todos los comandos se corren desde la raíz del repo.

1. **Descargar** Chula completo (copia en Roboflow `parasites-onxa6/parasites-trgfz`,
   versión 1, formato YOLOv8, CC BY 4.0, ~470 MB). Solo necesita `bash`, `curl` y
   `python3`; en Apolo se corre desde el nodo de login:
   ```bash
   export ROBOFLOW_API_KEY=...   # nunca se commitea
   bash data_prep/src/download_chula.sh data/roboflow
   ```
   El script no sobrescribe: si `data/roboflow` ya tiene archivos, se detiene. Descarga y
   descomprime en una carpeta temporal, verifica que estén `data.yaml` y los tres splits,
   y solo entonces la mueve a `data/roboflow`. Al final muestra las clases y cuántas fotos
   hay por split de Roboflow (ese split no se usa; el nuestro sale del paso 2).
2. **Filtrar a 3 clases y particionar** (70/15/15 por foto original, seed 42; YOLO +
   COCO; solo usa la librería estándar):
   ```bash
   python3 data_prep/src/filter_classes.py --src data/roboflow --dst data/raw
   ```
   Resultado esperado — fotos por split:

   | | ascaris | hookworm | trichuris |
   |---|---|---|---|
   | train | 631 | 643 | 638 |
   | valid | 135 | 138 | 137 |
   | test | 136 | 137 | 136 |

   Si salen otros números, el split no es el del equipo. `data/raw/data.yaml` sirve
   directo para YOLO (`nc: 3`); cada split trae `_annotations.coco.json` (ids COCO
   1=ascaris, 2=hookworm, 3=trichuris).
3. **Recortar cada huevo** (margen 15 %, formato ImageFolder + `crops.csv`; necesita
   `pillow` y `pyyaml` de `requirements.txt`):
   ```bash
   python3 data_prep/src/crop_eggs.py --raw data/raw --out data/crops --margin 0.15
   ```
4. **Embeddings DINOv2** (rama ML):
   ```bash
   python3 ml/src/extract_embeddings.py --crops-csv data/crops/crops.csv \
     --model dinov2_vitb14 --out-dir ml/outputs/embeddings
   ```
   → `ml/outputs/embeddings/dinov2_vitb14_224.npz` con `cls` y `patch_mean`, alineados
   con `labels`, `splits` y `groups` (foto original, para `StratifiedGroupKFold`).

**Análisis exploratorio:** [`data_prep/notebooks/eda_dataset.ipynb`](data_prep/notebooks/eda_dataset.ipynb) (después de
los pasos 1 y 2; sin GPU, ~2 min en el nodo de login de Apolo):
```bash
jupyter nbconvert --to notebook --execute --inplace data_prep/notebooks/eda_dataset.ipynb
```

## Reglas

- Detector, rama ML y rama DL usan **el mismo split**; `test` no se toca hasta el final.
- Datos, pesos y claves **nunca** se suben al repo.
- Las decisiones nuevas se registran en `docs/PLAN.md` (§14).
