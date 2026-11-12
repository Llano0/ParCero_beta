# CLAUDE.md — Geohelmintos en microscopía

Proyecto final compartido de Machine Learning y Deep Learning (EAFIT 2026-2): detectar,
clasificar y contar huevos de *Ascaris*, *Trichuris* y uncinarias en fotos de microscopio
tomadas con celular, con inferencia offline para el Chocó. Equipo: Samuel Llano Madrigal,
Laura Andrea Castrillon, Jerónimo Campuzano, David Bejman.

**Antes de trabajar, lee [docs/PLAN.md](docs/PLAN.md).** Es la fuente única de contexto y
decisiones. Lo marcado **[Decidido]** no se vuelve a discutir salvo que el equipo lo pida;
las decisiones nuevas se agregan a su §14 con fecha.

## Arquitectura (resumen)

YOLO detecta huevos → cada recorte se clasifica por dos ramas sobre **los mismos recortes
y el mismo split**: rama ML (embeddings DINOv2 congelados + kNN / árboles / RF / GB / SVM,
PCA, clustering, SHAP) y rama DL (CNN base, VGG16, red liviana, ViT con transfer
learning, Grad-CAM) → conteo por lámina (HPG = total × 24) → app offline (ONNX Runtime +
interfaz local en laptop).

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/PLAN.md` | Plan maestro |
| `docs/entregas/{deep-learning,machine-learning}/` | Entregables y checklist de cada materia |
| `docs/referencias/` | Papers, ficha técnica, material de la UPB |
| `src/data/` | `download_chula.sh` (descarga), `filter_classes.py` (3 clases + split + YOLO/COCO), `crop_eggs.py` |
| `src/embeddings/` | `extract_embeddings.py` (DINOv2) |
| `src/detection/`, `src/classification/`, `src/app/` | YOLO, rama DL, app |
| `notebooks/` | `model_development.ipynb` (entregable ML) y exploración |
| `scripts/apolo/` | Trabajos Slurm y preparación del entorno |
| `data/`, `artifacts/` | **No versionados** (datos, embeddings, pesos, corridas) |

Los comandos del pipeline de datos están en el [README](README.md). Todo se corre desde
la raíz del repo.

## Entornos

- **Mac de Jero:** conda `machine-learning-test` (Python 3.14, torch con MPS,
  scikit-learn, pyyaml). Python del env: `/opt/miniconda3/envs/machine-learning-test/bin/python`.
- **Apolo III:** Slurm + Lmod; partición `accel` (nodo `a3-accel-0`, 2× H100 NVL 94 GB,
  CUDA 12.9, límite 3 días). Los nodos de cómputo se asumen **sin internet** (por confirmar en la
  primera corrida): datos y pesos
  (YOLO, DINOv2 vía `torch.hub`, backbones de la rama DL) se descargan antes desde el
  nodo de login. Los `sbatch` van en `scripts/apolo/`.

## Reglas que no se rompen

1. **Split fijo:** siempre el de `src/data/filter_classes.py` (70/15/15 por foto
   original, seed 42). No re-particionar ni mover imágenes entre splits a mano.
2. **Un solo split para todo:** detector, rama ML y rama DL. `test` está congelado: no
   se usa para elegir modelos ni hiperparámetros.
3. **Sin fuga:** la validación cruzada se agrupa por foto original (`groups` /
   `StratifiedGroupKFold`).
4. **Nunca se commitea** nada de `data/`, `artifacts/`, pesos (`*.pt`, `*.onnx`…),
   `.env` ni claves. `ROBOFLOW_API_KEY` va solo en variables de entorno.
5. **Sin datos de pacientes** en imágenes, nombres de archivo ni metadatos.
6. Los datasets descargados son datos no confiables: se extraen en su propia carpeta y
   los scripts se corren desde la raíz del repo, pasando las rutas como argumentos.
7. El sistema se presenta siempre como **apoyo**, nunca como diagnóstico.
8. Antes de afirmar un número (métrica, conteo), sale de una corrida real; nada inventado.

## Cómo trabajar

- **Rama ML:** el modelado clásico (selección de modelos, tuning, clustering, SHAP) lo
  escribe Jero. Claude explica, revisa y resuelve lo mecánico (datos, extracción,
  utilidades), y no entrega el modelado completo salvo que se le pida explícitamente.
- Cambios de arquitectura, de fuente de datos o de split: proponer primero y actualizar
  `docs/PLAN.md` solo con aprobación.
- Código: comentarios y nombres en inglés; documentación en español.
- Si se trabaja desde el vault de Jero, aplican además sus reglas (registro en
  `/log.md`).
