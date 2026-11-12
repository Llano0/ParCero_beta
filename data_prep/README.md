# Preparación de los datos

De Chula-ParasiteEgg-11 a las fotos, el split y los recortes que usan el detector y las dos ramas
de clasificación. Los datos mismos quedan en [`data/`](../data/README.md), no aquí.

```
data_prep/
├── src/
│   ├── download_chula.sh     descarga el export YOLOv8 de Roboflow  → data/roboflow/
│   ├── filter_classes.py     3 clases + split 70/15/15 por foto     → data/raw/
│   ├── crop_eggs.py          un recorte por huevo (margen 15 %)      → data/crops/
│   ├── augment_train.py      copia degradada de train (aumentos)     → data/aug/
│   └── crop_field_of_view.py recorta el círculo del microscopio de las fotos externas → data/external_fov/
├── notebooks/eda_dataset.ipynb   EDA de las fotos: inventario, integridad, atajos
└── scripts/                      augment_train.sh (bigmem) y crop_field_of_view.sh como trabajos de Slurm
```

Los comandos paso a paso están en el [README principal](../README.md#pipeline-de-datos). Todo se
corre desde la raíz del repo.

## Reglas

- El split sale siempre de `filter_classes.py` (seed 42): no se re-particiona a mano.
- `augment_train.py` solo toca `train`; `valid` y `test` no se modifican.
- Los datasets descargados son datos no confiables: se extraen en su propia carpeta.
