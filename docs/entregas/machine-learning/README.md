# Entregas — Fundamentos de Machine Learning

Prof. Yomin Estiven Jaramillo Munera. Nota: entrega oportuna 50 % + presentación final 50 %.
Enunciado: `Requeriments_FinalP.pdf` (repo del curso, carpeta `FinalBoss`).

| Entregable | Fecha | Estado | Dónde |
|---|---|---|---|
| Entrega completa (notebook, repo, video) | **3-nov-2026** | Notebook y app listos; video por grabar | — |
| Video demo (2 min: modelo desplegado + interfaz) | 3-nov-2026 | Guion listo | [`guion-video.md`](guion-video.md) · enlace aquí |
| Repositorio (documentación y justificación de decisiones) | 3-nov-2026 | Listo | https://github.com/JeronimoCampuzanoC/scrambled-eggs |
| `model_development.ipynb` | 3-nov-2026 | Listo, ejecutado con salidas | [`notebooks/model_development.ipynb`](../../../notebooks/model_development.ipynb) |
| Presentación final | 19-nov-2026 | — | `presentacion/` |

## Checklist de requisitos (sección del notebook)

- [x] **3.1 Problema:** contexto, usuarios, decisión que apoya, origen de los datos (§1)
- [x] **3.2 Preprocesamiento** documentado y justificado: partición por foto, recorte, relleno a cuadrado,
      DINOv2, escalado y PCA dentro del `Pipeline`, faltantes, atípicos (§2)
- [x] **3.3 EDA profundo** con insights y tabla insight → decisión (§3; detalle en `eda_dataset.ipynb` y
      `eda_embeddings.ipynb`)
- [x] **3.4 Paradigma:** no supervisado (PCA, K-Means, jerárquico, DBSCAN; §4) + supervisado (kNN, árbol, RF,
      Gradient Boosting, SVM lineal y RBF; §5)
- [x] **3.5 Metodología y selección de modelo:** 160 configuraciones con `GridSearchCV` +
      `StratifiedGroupKFold`, regla de decisión, sensibilidad a hiperparámetros (§5)
- [x] **3.6 Generalización:** brecha train/CV, curva de aprendizaje, fuga, detector de atajos, robustez a
      perturbaciones, de punta a punta con YOLO, `test` una sola vez (§6 y §8)
- [x] **3.7 Métricas** justificadas en términos del uso real, bootstrap, rechazo, conteo por foto (§7)
- [x] **3.8 Despliegue + interfaz** offline en español (`src/app/app.py`; §10)
- [x] **3.9 Estrategia de reajuste:** monitoreo sin etiquetas demostrado, disparadores, validación campeón/retador,
      puesta en producción (§11)
- [x] Explicabilidad con **SHAP** (§9)

## Pendiente

- Grabar y subir el video (guion en [`guion-video.md`](guion-video.md)) y pegar el enlace arriba.
- Fotos externas de la UPB y de campo para medir la caída real (criterio: ≤ 10 puntos de F1).
- Presentación final (19-nov): problema, metodología, selección del modelo, generalización, métricas,
  despliegue y reajuste.
