# data/ (no versionado)

Todo lo que hay aquí se descarga o se genera con los scripts; git solo guarda este
archivo. Cada integrante y cada máquina (Mac, Apolo) lo reconstruye igual.

```
data/
├── roboflow/        export YOLOv8 de Chula completo (11 clases) ← data_prep/src/download_chula.sh
├── raw/             3 clases + split fijo (YOLO + COCO)       ← data_prep/src/filter_classes.py
├── crops/           un recorte por huevo (ImageFolder) + crops.csv ← data_prep/src/crop_eggs.py
└── external/        fotos externas (UPB / campo), solo para prueba final
```

- El split se genera siempre con `filter_classes.py` (seed 42): **no se re-particiona a mano.**
- `external/` no contiene ningún dato de pacientes.
