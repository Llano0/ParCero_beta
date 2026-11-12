# Geohelmintos en microscopía — detección, clasificación y conteo

Sistema de apoyo al coprológico para comunidades del Chocó: a partir de fotos de
microscopio tomadas con celular, detecta, clasifica y cuenta huevos de *Ascaris
lumbricoides*, *Trichuris trichiura* y uncinarias, sin necesidad de internet.

Proyecto final de **Machine Learning** y **Redes Neuronales y Deep Learning** (EAFIT
2026-2), en alianza con Microbiología y Parasitología de la UPB. Equipo: Samuel Llano
Madrigal, Laura Andrea Castrillon, Jerónimo Campuzano y David Bejman.

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
- **Nota macOS/conda:** torch (pip) y scikit-learn (conda) traen dos copias de OpenMP. En un mismo proceso
  que ya usó KMeans o t-SNE, cargar DINOv2 se cuelga; `model_development.ipynb` fija `OMP_NUM_THREADS=1` al
  inicio por eso.

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

## Rama ML: modelo y app (entrega de Machine Learning)

Todo corre en una laptop, sin GPU y sin internet una vez descargados los datos y los pesos.

1. **Pesos del detector** (YOLO11n de una clase, entrenado en Apolo; no se versionan): copiarlos en
   `artifacts/yolo/h_single_aug/weights/best.pt`. La elección del detector está en
   [`notebooks/compare_detectors.ipynb`](notebooks/compare_detectors.ipynb).
2. **Preparar recortes y embeddings** (pasos 2–4 de arriba, más los recortes del detector en `valid` y `test`;
   salta lo que ya existe):
   ```bash
   bash scripts/local/prepare_ml.sh
   ```
3. **Entrenar, evaluar y exportar el modelo:**
   [`notebooks/model_development.ipynb`](notebooks/model_development.ipynb), el entregable del curso (EDA,
   no supervisado, selección del modelo, generalización, métricas, SHAP, despliegue y reajuste). Guarda el
   paquete del modelo en `artifacts/ml/egg_classifier.joblib`. ~2 min:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/model_development.ipynb
   ```
4. **App offline** (Gradio, en español, para personal sin formación técnica):
   ```bash
   python src/app/app.py          # http://127.0.0.1:7860
   python src/app/app.py --lan    # también desde un celular conectado al hotspot de la laptop
   ```
   Foto → cajas con especie y confianza → aviso "consulte a un experto" en los dudosos → conteo por lámina,
   HPG estimado (× 24) e intensidad OMS. No guarda las fotos; registra las señales del modelo en
   `artifacts/app_logs/predictions.csv` para el monitoreo de *drift*.

   **Muestras (Kato-Katz, dos láminas por muestra).** La app guarda cada muestra en una base SQLite local
   (`artifacts/app_data/muestras.db`, no versionada; `--db otra.db` usa otra, p. ej. para pruebas). Se crea
   la muestra con un **código anónimo** (nunca el nombre del paciente), se marca **Lámina 1** o **Lámina 2** y
   cada foto analizada se suma a esa lámina. Resultado por especie: huevos de cada lámina, promedio,
   **HPG = promedio × 24** e intensidad OMS; si solo hay una lámina, el HPG usa esa y la app lo avisa. Los
   huevos dudosos quedan "por revisar" y no entran al conteo. Una foto no se puede sumar dos veces a la misma
   muestra. El historial se exporta a CSV desde la app. No se guardan fotos: solo un *hash* de cada foto y la
   especie y señales de sus huevos. Código en [`src/app/samples_db.py`](src/app/samples_db.py).

| Resultado (sale del notebook) | |
|---|---|
| Modelo | SVM lineal sobre embeddings DINOv2 ViT-B/14 |
| F1 macro: CV agrupada / `valid` / `test` | 0,990 / 0,990 / 0,998 |
| Conteo exacto por foto en `test` (YOLO + SVM) | 96,6 % de 409 fotos |
| Tiempo por foto en CPU | ~90 ms |

## Reglas

- Detector, rama ML y rama DL usan **el mismo split**; `test` no se toca hasta el final.
- Datos, pesos y claves **nunca** se suben al repo.
- Las decisiones nuevas se registran en `docs/PLAN.md` (§14).
