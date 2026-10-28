# data/ (no versionado)

Todo lo que hay aquí se descarga o se genera con los scripts; git solo guarda este
archivo. Cada integrante y cada máquina (Mac, Apolo) lo reconstruye igual.

```
data/
├── roboflow/        export YOLOv8 de Chula completo (11 clases) — descarga, ver README raíz
├── raw/             3 clases + split fijo (YOLO + COCO)       ← src/data/filter_classes.py
├── crops/           un recorte por huevo (ImageFolder) + crops.csv ← src/data/crop_eggs.py
└── external/        fotos externas (UPB / campo), solo para prueba final
```

- El split se genera siempre con `filter_classes.py` (seed 42): **no se re-particiona a mano.**
- `external/` no contiene ningún dato de pacientes.
