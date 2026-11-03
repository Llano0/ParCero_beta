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

```
├── docs/
│   ├── PLAN.md                 plan maestro (fuente única de contexto y decisiones)
│   ├── entregas/
│   │   ├── deep-learning/      correo de objetivos, informe, presentación
│   │   └── machine-learning/   notebook, video, presentación
│   └── referencias/            papers, ficha técnica, material de la UPB
├── src/
│   ├── data/                   download_chula.sh → filter_classes.py → crop_eggs.py
│   ├── embeddings/             extracción DINOv2 (rama ML)
│   ├── detection/              YOLO
│   ├── classification/         CNN / ViT (rama DL)
│   └── app/                    app offline (ONNX + interfaz local)
├── notebooks/                  model_development.ipynb (entregable ML) y exploración
├── scripts/apolo/              trabajos Slurm y preparación del entorno en Apolo III
├── data/                       (no versionado) ver data/README.md
└── artifacts/                  (no versionado) embeddings, pesos, corridas, exportes
```

## Entorno

- **Mac (desarrollo):** conda `machine-learning-test` (Python 3.14, torch con MPS).
- **Apolo III (entrenamiento):** partición `accel` (2× H100). Los nodos de cómputo no
  tienen internet: descargar datos y pesos desde el nodo de login.
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
   bash src/data/download_chula.sh data/roboflow
   ```
   El script no sobrescribe: si `data/roboflow` ya tiene archivos, se detiene. Descarga y
   descomprime en una carpeta temporal, verifica que estén `data.yaml` y los tres splits,
   y solo entonces la mueve a `data/roboflow`. Al final muestra las clases y cuántas fotos
   hay por split de Roboflow (ese split no se usa; el nuestro sale del paso 2).
2. **Filtrar a 3 clases y particionar** (70/15/15 por foto original, seed 42; YOLO +
   COCO; solo usa la librería estándar):
   ```bash
   python3 src/data/filter_classes.py --src data/roboflow --dst data/raw
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
   python3 src/data/crop_eggs.py --raw data/raw --out data/crops --margin 0.15
   ```
4. **Embeddings DINOv2** (rama ML):
   ```bash
   python3 src/embeddings/extract_embeddings.py --crops-csv data/crops/crops.csv \
     --model dinov2_vitb14 --out-dir artifacts/embeddings
   ```
   → `artifacts/embeddings/dinov2_vitb14_224.npz` con `cls` y `patch_mean`, alineados
   con `labels`, `splits` y `groups` (foto original, para `StratifiedGroupKFold`).

**Análisis exploratorio:** [`notebooks/eda_dataset.ipynb`](notebooks/eda_dataset.ipynb) (después de
los pasos 1 y 2; sin GPU, ~2 min en el nodo de login de Apolo):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_dataset.ipynb
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
