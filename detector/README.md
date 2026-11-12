# Detector YOLO

Entrena el detector de huevos (YOLO11n), recorta lo que detecta para las ramas ML y DL y compara
los modelos entrenados. Sigue en revisión: el detector de una sola clase (`single_cls`) con
aumentos es el candidato actual.

```
detector/
├── src/detect_crops.py                  corre el detector y recorta cada caja como crop_eggs.py
├── notebooks/compare_detectors.ipynb    sin aumentos contra con aumentos, en fotos externas y en valid
├── scripts/
│   ├── train_yolo.sh                    entrenamiento en GPU (accel)
│   ├── train_yolo_cpu.sh                el mismo en CPU (bigmem)
│   └── run_compare_detectors.sh         ejecuta el notebook de comparación en un nodo
└── outputs/                             (no versionado)
    ├── yolo/                            una carpeta por corrida (pesos, curvas, results.csv)
    ├── weights/                         pesos base descargados (yolo11n.pt)
    ├── predict/                         pruebas con fotos sueltas
    └── detected/                        recortes de lo que detecta YOLO (valid, external)
```

## Cómo se entrena

Siempre desde la raíz del repo; lo que se escribe después del script se pasa a YOLO.

```bash
sbatch --time=00:30:00 detector/scripts/train_yolo.sh epochs=100 patience=20 name=mi_corrida single_cls=True
```

Los resultados quedan en `detector/outputs/yolo/<name>/` y el modelo en `weights/best.pt`. Los
logs, en `logs/detector/`.

## Reglas

- Mismo split que todo el proyecto; las corridas se comparan en `valid`, nunca en `test`.
- Los pesos (`*.pt`) no se suben a git.
