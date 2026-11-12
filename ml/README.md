# Rama de Machine Learning

Todo lo propio de la rama ML: embeddings de DINOv2, su análisis exploratorio y los scripts para
generarlos. El modelado clásico (kNN, árboles, RF, GB, SVM, PCA, clustering, SHAP) y el
`model_development.ipynb` también van aquí. Entregables y checklist de la materia:
[`docs/entregas/machine-learning/`](../docs/entregas/machine-learning/README.md).

```
ml/
├── src/extract_embeddings.py        recortes → embeddings DINOv2 (CLS + media de parches)
├── notebooks/eda_embeddings.ipynb   EDA de los embeddings
├── scripts/
│   ├── prepare_ml.sh                recortes y embeddings (a mano y de YOLO), en bigmem
│   └── run_eda_embeddings.sh        ejecuta el notebook de EDA en un nodo
└── outputs/                         (no versionado) .npz de embeddings
    ├── embeddings/                  recortes dibujados a mano (train, valid, test)
    └── embeddings_detected/         recortes de YOLO (valid)
```

## Lo que consume de las otras carpetas

| Qué | De dónde |
|---|---|
| Fotos, split y recortes a mano | `data/` (`data_prep/` los genera) |
| Recortes de lo que detecta YOLO | `detector/outputs/detected/` |
| Pesos del detector | `detector/outputs/yolo/` |

## Cómo correrlo

Siempre desde la raíz del repo, con el ambiente `geohelmintos`.

```bash
sbatch ml/scripts/prepare_ml.sh          # recortes + embeddings (usa el detector h_single_aug)
sbatch ml/scripts/run_eda_embeddings.sh  # ejecuta ml/notebooks/eda_embeddings.ipynb
```

`prepare_ml.sh` se detiene si `data/crops` o alguna carpeta de `detector/outputs/detected` ya
existe: se borra y se vuelve a enviar. Los logs quedan en `logs/ml/`.

## Reglas

- Mismo split que todo el proyecto; `test` congelado, sin tocarlo hasta el final.
- Escalado y PCA se ajustan solo con `train`; la validación cruzada se agrupa por foto original.
- Nada de `ml/outputs/` se sube a git.
