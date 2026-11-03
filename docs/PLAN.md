# Plan maestro — Detección, clasificación y conteo de geohelmintos en imágenes de microscopio

> Proyecto final compartido entre **Machine Learning** (prof. Yomin Estiven Jaramillo
> Munera) y **Redes Neuronales y Deep Learning** (prof. Raúl Andrés Castañeda
> Quintero), EAFIT 2026-2, en alianza con el área de Microbiología y Parasitología de la
> **Universidad Pontificia Bolivariana (UPB)**.
>
> Documento vivo y **fuente única** de contexto y decisiones. Integra la *Ficha técnica*
> preliminar del equipo (oct-2026); donde hay diferencias, prevalece lo decidido aquí
> (ver §15). Última actualización: 2026-11-02.
>
> Convención: **[Decidido]** acordado por el equipo · **[Propuesta]** sugerido, por
> confirmar · **[De la ficha]** aportado por la ficha técnica, incorporado ·
> **[Decisión de Claude]** criterio tomado para no bloquear, revisable ·
> **[Pendiente]** pregunta abierta (§16).

---

## 1. Contexto y problema

Las **geohelmintiasis** (parásitos transmitidos por el suelo) afectan a unos **1.500
millones de personas (24 % de la población mundial)**, sobre todo en comunidades pobres
con poco acceso a agua y saneamiento (OMS). En el **Chocó** son muy frecuentes: en un
estudio de 2021 en dos comunidades indígenas, los helmintos más frecuentes en niños fueron
*Trichuris* (75 %) y *Ascaris* (70,4 %) (*Enfermedades Infecciosas y Microbiología
Clínica*, 2023) **[De la ficha — verificar citas]**.

El diagnóstico se hace con un **coprológico**: se prepara una lámina con la muestra de
heces y un microscopista la recorre buscando huevos, que identifica por su **morfología**
(forma, cáscara, tapones, contenido), no por su tamaño en la foto (depende del aumento).

Problemas del flujo actual en campo:

- Reconocerlos exige un microscopista entrenado que muchas comunidades no tienen cerca;
  identificar y contar es lento y propenso a error por fatiga.
- El método estándar de la OMS para estos parásitos (**Kato-Katz**) exige contar
  **todos** los huevos de la lámina para estimar la intensidad de la infección.
- En la mayoría de zonas del Chocó **no hay conexión a internet**: la herramienta tiene
  que funcionar **offline**.

**Usuarios:** microscopistas, bacteriólogos o personal de salud no especializado en
brigadas de campo, con microscopio, **celular** (fotos por el ocular) y una **laptop** en
el campamento.

**Qué apoya el sistema:** detectar, identificar y contar huevos por especie en cada foto,
llevar el conteo por lámina y estimar la intensidad de infección, con un nivel de
confianza y aviso de "consulte a un experto" cuando duda. **Es apoyo, nunca diagnóstico**:
la confirmación la hace un profesional.

### 1.1 Flujo de laboratorio de referencia

1. **Recolección:** heces en coprocolector limpio y seco, rotulado; procesar el mismo día
   o fijar (formol 10 %) si hay demora. A veces 3 muestras seriadas (eliminación
   intermitente de huevos).
2. **Macroscópico:** consistencia, color, moco, sangre, adultos visibles.
3. **Preparación de la lámina** (cambia la apariencia de la imagen):
   - *Examen directo:* aplicador de madera + solución salina; a menudo una segunda
     preparación con **lugol** (tiñe café-amarillo).
   - *Concentración:* sedimentación / flotación (más sensible).
   - ***Kato-Katz:*** plantilla de **41,7 mg**, celofán con glicerina y verde de
     malaquita (fondo verdoso). Se cuentan todos los huevos → **HPG = conteo × 24**. Las
     uncinarias se aclaran y desaparecen a los **30–60 min**: leer rápido.
4. **Lectura:** recorrido en zigzag con **10x** (≈150–300 campos por lámina de Kato-Katz),
   confirmación con **40x** (≈3500–5000 campos si se recorriera toda).

---

## 2. Objetivos

### 2.1 Entrega preliminar de DL (correo, **jueves 29-oct-2026**) **[Propuesta]**

- **Para:** prof. Raúl Andrés Castañeda Quintero.
- **Asunto:** `Proyecto Final – Grupo #XX – Objetivos del proyecto` **[Pendiente: #]**.

**Título.** Detección, clasificación y conteo de geohelmintos (*Ascaris*, *Trichuris* y
uncinarias) en imágenes de microscopio mediante Deep Learning, como apoyo al diagnóstico
en comunidades del Chocó.

**Objetivo general.** Desarrollar y evaluar un sistema basado en Deep Learning que, a
partir de fotografías de microscopio tomadas con celular, detecte, clasifique y cuente
huevos de *Ascaris lumbricoides*, *Trichuris trichiura* y uncinarias, que funcione sin
conexión a internet y que mantenga su desempeño con imágenes de fuentes distintas a las de
entrenamiento.

**Objetivos específicos.**

1. Construir un conjunto de datos reproducible y sin fuga (filtrado de clases, corrección
   de artefactos, partición por foto original en entrenamiento, validación y prueba,
   aumento de datos y diccionario de datos) y entrenar un detector de huevos YOLO con
   transfer learning.
2. Entrenar y comparar clasificadores de huevos sobre los mismos recortes —una CNN base,
   VGG16 y una red liviana (EfficientNet-B0 / MobileNetV3) con transfer learning, y un
   Vision Transformer—, reportando F1 macro, matriz de confusión y desempeño en imágenes
   externas.
3. Evaluar la robustez e interpretabilidad del mejor modelo (perturbaciones de brillo y
   nitidez, Grad-CAM) e integrarlo en una aplicación offline que lleve el conteo por
   lámina y estime la intensidad de infección.

### 2.2 Objetivos del proyecto completo

Los anteriores, más el componente de Machine Learning: clasificar los mismos recortes con
**embeddings de DINOv2 + modelos clásicos** (§6.2), compararlos con la rama DL, desplegar
el sistema y diseñar su estrategia de reajuste.

### 2.3 Criterios de éxito **[De la ficha — a acordar con la UPB]**

- **F1 macro ≥ 0,85** en la prueba interna (partición por foto original).
- **Caída ≤ 10 puntos de F1** entre la prueba interna y las imágenes externas (UPB/campo).
- **Grad-CAM revisado por un parasitólogo** en una muestra de aciertos y errores.
- Punto de decisión: si la caída con imágenes externas supera 10 puntos, se vuelve a los
  datos antes de pasar a campo.

---

## 3. Alcance

| | |
|---|---|
| **Dentro** | Detección + clasificación de 3 especies en fotos de campo; conteo por foto y por lámina; HPG + intensidad OMS como **estimación de apoyo**; confianza + rechazo ("consulte a un experto"); app offline en laptop; comparación rama DL vs rama ML. |
| **Extra (si hay tiempo)** | Clase "otro / sin parásito" en el clasificador (§6.5); app en celular; aviso de calidad de imagen; evaluación de conteo con mosaicos; segmentación SAM + U-Net; estadio de *Ascaris* (fértil/infértil). |
| **Fase posterior** | *Strongyloides stercoralis* como cuarta clase (son **larvas**, no huevos; hay que compararla contra larvas de uncinaria) cuando haya imágenes suficientes de la UPB. |
| **Fuera** | Diagnóstico clínico autónomo; uso sin supervisión; certificación regulatoria; escaneo automático de la lámina completa (requiere platina motorizada). |
| **Anti-objetivos** | Métricas infladas por fuga, duplicados o atajos de fuente; un modelo que solo funcione con las imágenes de entrenamiento; guardar datos de pacientes; depender de internet en la inferencia. |

---

## 4. Clases objetivo **[Decidido]**

| Clase (id YOLO / id COCO) | Especie | Rasgos morfológicos clave |
|---|---|---|
| `ascaris` (0 / 1) | *Ascaris lumbricoides* | Ovalado-redondo, cáscara gruesa y mamelonada, café; formas fértil, infértil y decorticada. |
| `hookworm` (1 / 2) | Uncinarias (*Necator americanus*, *Ancylostoma duodenale*) | Ovalado, cáscara **delgada y transparente**, espacio claro alrededor de los blastómeros. |
| `trichuris` (2 / 3) | *Trichuris trichiura* | **Barril / limón** con **tapones polares**, café-amarillo. |

Son los **geohelmintos** que Kato-Katz cuantifica. Chula trae otras 8 especies
(Capillaria, Enterobius, Fasciolopsis, Hymenolepis diminuta y nana, Opisthorchis,
Paragonimus, Taenia), útiles como clase "otro" (§6.5).

---

## 5. Datos

### 5.1 Fuentes

| Fuente | Origen | Licencia | Uso |
|---|---|---|---|
| **Chula-ParasiteEgg-11** | Universidad Chulalongkorn, Tailandia (ICIP 2022) | CC BY 4.0 | Base de entrenamiento (3 clases) |
| **Copia completa en Roboflow** `parasites-onxa6/parasites-trgfz` v1 | Chula completo: 10 000 fotos, 11 clases, ~910 huevos por clase | CC BY 4.0 | **Fuente decidida** (§5.2) |
| Copia parcial en Roboflow `research-t0os5/test-parasite-egg-detection` v4 | Subconjunto (~200 huevos por clase) | CC BY 4.0 | Primera fuente; reemplazada (§5.4) |
| ParasitoBank | Universidad del Cauca, Colombia | CC BY 4.0 | Strongyloides (fase posterior) |
| CDC DPDx | CDC, EE. UU. | Dominio público, citar | Strongyloides / externas |
| Atlas LACES | Universidad Federal de Goiás, Brasil | No especificada: pedir permiso fuera del curso | Strongyloides |
| **UPB** + fotos de campo (§5.7) | Alianza / amiga del equipo | Uso acordado; sin datos de pacientes | **Solo prueba externa** |

### 5.2 Fuente decidida: Chula completo (`parasites-trgfz` v1) **[Decidido 2026-10-27]**

| | `test-parasite-egg-detection` v4 (anterior) | **`parasites-trgfz` v1 (actual)** |
|---|---|---|
| Nuestras 3 clases | 603 fotos, 818 cajas (con duplicados) | **2731 fotos, 2749 huevos** (Ascaris 910, uncinarias 928, Trichuris 911) |
| Huevos por foto | Parecían hasta 5; en realidad casi siempre 1 (reflejos) | 1 (solo 18 fotos con 2) |
| Duplicados por reflejo | 160 pares | 0 |
| Nombres originales de Chula | No (`2005_jpg.rf.…`) | **Sí** (`Ascaris-lumbricoides_0110_jpg.rf.…`): agrupar por foto original es directo |
| Preprocesamiento | 640 "Fit (reflect edges)" | 640 **"Stretch to"** (deforma) |
| Split de Roboflow | Todo en `valid` | train 1930 / valid 524 / test 277 (no se usa: aplicamos el nuestro, §5.5) |

Es el mismo origen que el "dataset v5" de la ficha (~900 por clase), pero con **fotos
completas y cajas**: detector y ambas ramas salen de la misma fuente, sin riesgo de fuga
entre ellos. Formato de descarga: **YOLOv8** (el filtrado genera también COCO).

**Deformación por estiramiento.** Como los huevos aparecen en cualquier orientación, el
promedio de log(ancho/alto) de las cajas debería ser ≈ 0. En v4 da +0,01; en esta
versión da −0,07: en promedio las fotos quedaron **comprimidas ~7 % en horizontal**. No
se puede corregir foto por foto porque no se conoce su tamaño original. Es leve frente a
la diferencia de forma entre especies, pero se reporta como limitación. **[Propuesta]**
Para eliminarla: hacer *fork* del proyecto en el workspace del equipo y generar una
versión sin redimensionar (o "Fit (black edges)"), que además daría más resolución.

### 5.3 Pipeline de datos

Descarga (detalles en el README):

```bash
export ROBOFLOW_API_KEY=...          # nunca se commitea
bash src/data/download_chula.sh data/roboflow   # ~470 MB
```

Flujo:

```
   │  src/data/download_chula.sh   descarga del export YOLOv8 de Roboflow
   ▼
data/roboflow/             export completo (11 clases, 10 000 fotos)
   │  src/data/filter_classes.py   3 clases, ids 0..2, split 70/15/15 por foto, seed 42
   ▼
data/raw/                  YOLO (data.yaml, nc 3) + COCO (<split>/_annotations.coco.json)
   │  src/data/crop_eggs.py        un recorte por huevo, margen 15 %
   ▼
data/crops/<split>/<clase>/*.png + crops.csv     (formato ImageFolder)
   │  src/embeddings/extract_embeddings.py   DINOv2 → CLS + media de parches
   ▼
artifacts/embeddings/<modelo>_224.npz     cls, patch_mean, labels, splits, groups, crop_paths
```

Split resultante (verificado: 2731 fotos, 2749 cajas; COCO con 1924 / 413 / 412 cajas):

| Fotos por split | ascaris | hookworm | trichuris |
|---|---|---|---|
| train | 631 | 643 | 638 |
| valid | 135 | 138 | 137 |
| test | 136 | 137 | 136 |

### 5.4 Historia: la primera fuente y el hallazgo del reflejo

La primera fuente fue la copia parcial `test-parasite-egg-detection` v4. Sus versiones
v3/v7 recortan al centro y pierden 144 de 603 fotos. La v4 usa "Fit (reflect edges)", que
refleja la foto **y las cajas** en el relleno: de las 167 fotos con varias cajas, 142
tenían el mismo huevo reflejado (confirmado visualmente). Con la fuente nueva el problema
desaparece. Igual es un buen hallazgo de EDA para ambas presentaciones: un artefacto de
preprocesamiento que inflaba muestras y conteos, detectado y documentado antes de
entrenar.

### 5.5 Reglas de partición **[Decidido]**

- Split **fijo**, estratificado por clase (y por fuente cuando haya varias), **agrupado
  por foto original**: todos los huevos de una foto quedan en el mismo split.
- **Detector, rama ML y rama DL usan el mismo split.** Si no, la comparación no vale.
- `test` **congelado**: no se usa para seleccionar modelos ni hiperparámetros.
- Validación cruzada de **5 pliegues** agrupada por foto (`StratifiedGroupKFold`) dentro
  de train(+valid).

### 5.6 Diccionario de datos **[De la ficha]**

Campos de `crops.csv` (los marcados con * ya existen; el resto se agrega):

| Campo | Tipo | Descripción |
|---|---|---|
| `crop_path`* | texto | Ruta del recorte |
| `label`* | categórica | ascaris, hookworm, trichuris (después strongyloides, otro) |
| `split`* | categórica | train, valid, test |
| `source_image`*, `group`* | texto | Foto original; define el grupo de partición |
| `x1, y1, x2, y2`*, `box_w`*, `box_h`* | número | Caja en la foto original (px) |
| `fuente` | categórica | Chula, ParasitoBank, CDC, UFG, UPB, campo |
| `aumento` | categórica | 100x, 400x, si se conoce |
| `tincion` | categórica | salina, lugol/yodo, Kato-Katz, sin teñir |
| `dispositivo` | categórica | Cámara o celular |
| `licencia` | categórica | CC BY 4.0, dominio público, … |
| `calidad` | categórica | buena, dudosa, descartada |
| `anotada` | booleano | Trae flechas o letras dibujadas (van aparte) |
| `verificada_por` | texto | Experto que confirmó la etiqueta |

**[Pendiente]** El profesor pidió explicar el significado de colores y marcas de las 8
imágenes anotadas: documentarlas con la leyenda de la fuente original.

### 5.7 Datos externos (prueba final) **[Propuesta]**

Dos orígenes: **imágenes de la UPB** y **fotos de campo con celular por el ocular**
(como la foto de referencia de la amiga del equipo: campo circular, mitad negra, puntero
del ocular, 1980×3520 vertical).

Protocolo de captura por lámina:

- Fotografiar **todos los campos con huevo** + **15–20 campos al azar sin huevo**
  (negativos: Chula no tiene imágenes vacías).
- Por foto: **cuántos huevos de cada especie** (confirmado por el experto; puede haber
  coinfección), aumento, preparación, celular.
- Por lámina: **conteo manual de referencia** por especie.
- **Cero datos de pacientes.** Si se usan imágenes de pacientes: consentimiento,
  anonimización y aprobación de un comité de ética.
- Meta mínima: 20–30 fotos con huevo + negativos. **Solo evaluación.**

**[Decisión de Claude] Verdad de referencia.** El equipo dibuja las cajas (Roboflow
Annotate o labelme) guiándose por el conteo del experto, que valida las dudosas. Con cajas
+ especie por huevo el set externo mide recall de detección, clasificación por huevo y
error de conteo.

### 5.8 Preprocesamiento y aumento de datos

**Preprocesamiento [Decidido]:** recorte rellenado **a cuadrado** (no estirado: la forma
es lo discriminativo) → 224 × 224 → normalización ImageNet. Exactamente el mismo flujo
para cualquier imagen externa.

**Experimentos [De la ficha]:** normalización de color por imagen; igualar la nitidez
entre fuentes reduciendo y reescalando las de mayor resolución.

**Aumento (solo entrenamiento):**

| Transformación | Variación que cubre |
|---|---|
| Rotación 0–360° y volteos | Bajo el microscopio no hay arriba ni abajo |
| Brillo, contraste, gamma | Luz y cámara |
| Tono y saturación (rango salina → lugol → fondo verdoso de Kato-Katz) | Tinción e iluminación |
| Desenfoque, ruido, compresión JPEG | Enfoque y calidad del celular |
| Zoom moderado / reducir y reescalar | Aumento del microscopio, nitidez entre fuentes |
| Viñeta circular con fondo negro, puntero de ocular sintético | Foto con celular por el ocular |

El tamaño en píxeles **no** es una feature confiable (σ ≈ 50 % de la media dentro de cada
clase en Roboflow).

---

## 6. Arquitectura **[Decidido: dos etapas con dos ramas de clasificación]**

```
              foto de microscopio (celular por el ocular)
                               │
        Etapa 1 — Detector YOLO (transfer learning desde COCO)
                               │ cajas  (sin cajas → "no se detectaron huevos")
                    recorte de cada huevo
               ┌───────────────┴────────────────┐
   Rama ML (Machine Learning)          Rama DL (Deep Learning)
   DINOv2 congelado → embedding        CNN base, VGG16, red liviana y ViT
   → escalado / PCA                    con transfer learning
   → kNN, árbol, RF, GB, SVM           → Grad-CAM
   → SHAP
               └───────────────┬────────────────┘
          clase por huevo + confianza (baja → "consulte a un experto")
                               │
        Agregación por lámina: conteo por especie, posibles duplicados,
        HPG = total × 24, intensidad OMS
                               │
                 App offline (laptop; celular como extra)
```

Por qué dos etapas y no solo YOLO (la pregunta que hará el profesor de DL): separa
*dónde* de *qué*. El clasificador ve el huevo recortado y ampliado, no una parte pequeña
de una foto de 640 px. El detector puede entrenarse con más datos (las 11 clases como
"huevo"). El clasificador se cambia sin reentrenar el detector. Se demuestra con números
contra la cabeza de clasificación del propio YOLO (baseline).

### 6.1 Etapa 1 — Detector

- Ultralytics **YOLO11n / YOLOv8n** (≈6 MB, exportable a ONNX / TFLite), preentrenado en
  COCO, fine-tuning a 3 clases (`data/raw/data.yaml`). Tamaño *small* si nano no alcanza.
- La aumentación *mosaic* (por defecto) le muestra fotos con varios huevos aunque el
  dataset casi siempre tenga uno.
- "No hay parásito" = sin detecciones por encima del umbral; se calibra con los negativos
  del set externo.

### 6.2 Rama ML — embeddings + ML clásico (curso de Machine Learning)

- **Extractor:** DINOv2 **ViT-S/14** (384-d, liviano) y **ViT-B/14** (768-d). Se guardan
  CLS y media de parches. Implementado: `src/embeddings/extract_embeddings.py` (ViT-B/14 probado).
- **Implementado (2-nov):** `notebooks/model_development.ipynb`. SVM lineal, F1 macro 0,990 (CV y `valid`) y
  0,998 en `test`; conteo exacto en el 96,6 % de las fotos de `test` con YOLO + SVM.
- **Preprocesamiento:** `StandardScaler`, normalización L2, **PCA** (curva de varianza).
- **No supervisado:** PCA/t-SNE para visualizar; **K-Means**, **jerárquico**,
  **DBSCAN** (outliers / posibles errores de etiqueta); ARI y NMI contra las etiquetas.
  ¿Las especies se separan solas? ¿Aparecen subtipos (Ascaris fértil/infértil)?
- **Supervisado:** **kNN**, **árbol de decisión**, **Random Forest**, **Gradient
  Boosting**, **SVM** (lineal y RBF); `GridSearchCV` con `StratifiedGroupKFold`.
- **Explicabilidad:** **SHAP** sobre el modelo de árboles (componentes PCA y/o features
  interpretables como color y relación de aspecto).
- **Otros usos:** sugerir duplicados dentro de una lámina y detectar *drift* (§9).
- Aprendizaje por refuerzo: **no aplica**; no se fuerza.
- Diferencia con el ViT de la rama DL: aquí el ViT está **congelado** y el aprendizaje lo
  hacen los modelos clásicos; allá se **ajusta** la red completa.

### 6.3 Rama DL — CNN y ViT con transfer learning (curso de Deep Learning)

Modelos, entrenados todos con la misma partición:

| Modelo | Papel | Nota |
|---|---|---|
| CNN pequeña desde cero | Línea base | Mide cuánto aporta el preentrenamiento |
| **VGG16** con transfer learning | Pedido por los profesores | ~138 M parámetros: pesado para campo |
| **EfficientNet-B0 / MobileNetV3** (ResNet-50 como alternativa) | **Candidato para campo** | Elegir la más liviana que rinda casi igual a la mejor |
| **Vision Transformer** preentrenado | Recomendado por los profesores | No se espera que supere a la CNN con estos datos |

**Receta de partida [De la ficha]** (se ajusta y justifica con las curvas):

1. Entrada 224 × 224 rellenada a cuadrado, normalización ImageNet.
2. Fase 1: red congelada, solo la cabeza, 5–10 épocas, lr 1e-3.
3. Fase 2: descongelar los últimos bloques, lr 1e-4 → 1e-5 con programación coseno.
4. AdamW, *label smoothing* 0,1, lote 32, parada temprana por F1 de validación.
5. Tres semillas + CV agrupada por foto; reportar media y desviación.
6. Guardar pesos, configuración y partición de cada experimento.

**Interpretabilidad:** **Grad-CAM** sobre aciertos y errores, revisado por un
parasitólogo de la UPB.

La guía de DL exige además: referencia de cada arquitectura y modificaciones hechas,
curvas de loss y métricas, recursos computacionales, análisis de sobre/subajuste y
prueba con **muestras externas**.

### 6.4 Agregación y conteo por lámina **[Decidido: conteo asistido]**

El microscopista recorre la lámina como siempre (10x, zigzag) y toma foto cada vez que ve
un huevo u objeto sospechoso. La app:

1. Detecta y clasifica los huevos de la foto, con su confianza.
2. Suma al conteo de **esa lámina** (sesión).
3. **Sugiere** posibles **duplicados** (mismo huevo en campos solapados) por similitud de
   embeddings y **el usuario confirma**. **[Decisión de Claude]** No se automatiza:
   DINOv2 hace que huevos de la misma especie se parezcan, así que la similitud no
   distingue "el mismo huevo" de "otro huevo igual".
4. Al cerrar la lámina: **HPG = total × 24** por especie y categoría OMS, presentada
   como **estimación de apoyo**, no como resultado clínico.

Umbrales OMS en HPG **[verificar con la fuente OMS antes de citar]**:

| Especie | Leve | Moderada | Severa |
|---|---|---|---|
| *A. lumbricoides* | 1–4 999 | 5 000–49 999 | ≥ 50 000 |
| *T. trichiura* | 1–999 | 1 000–9 999 | ≥ 10 000 |
| Uncinarias | 1–1 999 | 2 000–3 999 | ≥ 4 000 |

Por qué no muestreo aleatorio de campos: con 2 huevos en la lámina y 20 % de campos
revisados, P(no ver ninguno) = 0,8² = 64 % → falso negativo en infecciones leves.

### 6.5 Clase "otro / sin parásito" **[De la ficha — extra prioritario]**

Para que el sistema pueda decir "no es ninguno de los tres" o "no hay parásito":

- En el **detector**: no detectar = sin parásito; se mide con los negativos externos.
- En el **clasificador**: clase "otro" con las 8 especies restantes de Chula y parches de
  fondo (revisar a mano que no tengan huevos sin marcar).
- Rechazo por **baja confianza** como respaldo.

---

## 7. Evaluación

| Qué | Métrica | Por qué |
|---|---|---|
| Detección | mAP@0.5, mAP@0.5:0.95, precisión y recall por clase | Recall pesa más: un huevo no detectado es un falso negativo clínico |
| Clasificación | **F1 macro**, recall por clase, matriz de confusión, exactitud balanceada, AUROC uno-contra-todos | Todas las clases importan por igual; Ascaris vs uncinaria es la confusión esperada |
| Confianza | Calibración y **curva de rechazo** (acierto si el modelo se abstiene cuando duda) | Sustenta el aviso "consulte a un experto" |
| Clustering | ARI, NMI, silueta | ¿La estructura natural coincide con las especies? |
| Conteo | MAE del conteo por lámina vs conteo manual; acuerdo en categoría OMS | Lo que usa el usuario real |
| Eficiencia | MB del modelo, ms por imagen en CPU | Restricción de campo |
| Generalización | Test congelado, CV agrupada, **por fuente**, **externas** | Sin sobreajuste, fuga ni atajos |
| Incertidumbre | Intervalos por *bootstrap* | Datos pequeños |

**Detector de atajos [De la ficha]:** un modelo de control que solo ve color y tamaño; si
acierta mucho, hay una pista equivocada. Reportar siempre por clase (y por fuente cuando
haya varias), nunca solo el total.

Conteo multi-huevo sin datos reales: mosaicos de 2–4 fotos **reales de prueba** con conteo
conocido (declarado como sintético).

---

## 8. Despliegue **[Decidido: laptop offline como entrega principal]**

- Modelos exportados a **ONNX** (YOLO nativo; CNN con `torch.onnx`; SVM/RF con
  `skl2onnx`) y ejecutados con **ONNX Runtime** → no requiere PyTorch en la laptop.
- Interfaz local **Gradio o Streamlit** (analítica desactivada), sin internet.
  Distribución por USB. **[Pendiente]** sistema operativo de la laptop de campo.
- **[Decisión de Claude] Del celular a la laptop sin internet:** la laptop crea un
  *hotspot* Wi-Fi y sirve la app en la red local; el celular abre la IP de la laptop y
  sube la foto con la cámara. Respaldo: copiar por cable.
- **Celular como extra** (requisito RNF1 de la ficha, pospuesto): PWA con
  `onnxruntime-web` o Android con TFLite; se mide tamaño y latencia en un gama media.

Flujo de la interfaz: *abrir lámina → foto → (aviso de calidad) → cajas, especie y
confianza → conteo acumulado → cerrar lámina → HPG + intensidad*. Selector de
clasificador (rama DL / rama ML) para la demo.

### 8.1 Requisitos **[De la ficha, ajustados]**

| Código | Requisito | Estado |
|---|---|---|
| RF1 | Detectar y clasificar Ascaris, Trichuris y uncinarias en una foto | Núcleo |
| RF2 | Mostrar la confianza y rechazar los dudosos ("consulte a un experto") | Núcleo |
| RF3 | Reconocer fotos sin parásito y otros parásitos | Extra prioritario (§6.5) |
| RF4 | Avisar si la foto está desenfocada u oscura | Extra |
| RF5 | Mostrar qué zona sustenta el resultado (Grad-CAM / cajas) | Núcleo (cajas); Grad-CAM extra en la app |
| RF6 | Conteo por lámina, HPG e intensidad OMS | Núcleo |
| RNF1 | Sin internet, modelo en el dispositivo | Núcleo en laptop; celular extra |
| RNF2 | Respuesta en pocos segundos por foto | Medir en CPU |
| RNF3 | Sin datos personales | Núcleo |
| RNF4 | Lenguaje claro para personas sin formación técnica | Núcleo |
| RNF5 | Versionar datos, código y modelos | Núcleo |
| RE1–RE3 | Citar fuentes y licencias; consentimiento y comité de ética si hay pacientes; siempre apoyo, nunca diagnóstico | Núcleo |

---

## 9. Estrategia de reajuste (requisito ML 3.9) **[Propuesta]**

Alcance de implementación **[Pendiente §16]**: si es demo del curso, se implementa el paso
1 y el resto se diseña y documenta.

1. **Monitoreo local offline:** la app guarda las fotos con baja confianza y las que
   quedan **lejos de la distribución de entrenamiento en el espacio DINOv2** (*drift*:
   nuevo microscopio, tinción o celular).
2. **Sincronización** cuando hay conexión (regreso a Quibdó / Medellín).
3. **Etiquetado** por un experto (UPB).
4. **Reentrenamiento en Apolo III** cuando el porcentaje de *drift* o de baja confianza
   supere un umbral, o se acumule un volumen mínimo de etiquetas nuevas.
5. **Validación** contra el test congelado + externas; se publica solo si no empeora.
6. **Distribución** versionada (USB / descarga cuando haya red), con tarjeta del modelo.

---

## 10. Infraestructura y reproducibilidad

| Dónde | Para qué | Detalles |
|---|---|---|
| Mac de Jero (M5 Pro, 24 GB) | Desarrollo, pipeline de datos, rama ML | conda `machine-learning-test` (Python 3.14, torch 2.14 + MPS, scikit-learn 1.9, pyyaml). Pesos de DINOv2 en `~/.cache/torch/hub`. |
| **Apolo III** (EAFIT) **[Decidido, en lugar de Colab]** | Entrenamiento de YOLO y rama DL, CV, barridos | Slurm + Lmod; partición **`accel`**, nodo `a3-accel-0`, **2× H100 NVL 94 GB**, CUDA 12.9, límite **3 días** por trabajo. Nodos de cómputo que se asumen **sin internet** (por confirmar en la primera corrida): descargar datos y pesos desde el nodo de login. |
| Laptop de campo (sin GPU) | Inferencia offline | ONNX Runtime + interfaz local. |

- **Framework [Decidido]:** PyTorch (+ Ultralytics para YOLO). Las H100 sobran para estos
  datos; su valor está en correr muchos experimentos en paralelo.
- **Reproducibilidad [De la ficha]:** semillas fijas, versionado de datos y modelos,
  seguimiento de experimentos (MLflow o Weights & Biases), **tarjeta del modelo** (uso
  previsto, datos, límites).

### 10.1 Repositorio

```
├── README.md                 cómo correr el pipeline
├── CLAUDE.md                 instrucciones operativas para Claude
├── requirements.txt
├── docs/
│   ├── PLAN.md               este documento
│   ├── entregas/
│   │   ├── deep-learning/    correo de objetivos, informe, presentación + checklist
│   │   └── machine-learning/ notebook, video, presentación + checklist
│   └── referencias/          papers, ficha técnica, material de la UPB
├── src/
│   ├── data/                 download_chula.sh, filter_classes.py, crop_eggs.py
│   ├── embeddings/           extract_embeddings.py (rama ML)
│   ├── detection/            YOLO
│   ├── classification/       rama DL
│   └── app/                  app offline
├── notebooks/                model_development.ipynb (entregable ML) y exploración
├── scripts/apolo/            trabajos Slurm, preparación del entorno, descarga de pesos
├── data/                     (no versionado) roboflow/, raw/, crops/, external/
└── artifacts/                (no versionado) embeddings, pesos, corridas, exportes
```

- **[Pendiente]** Crear el repo en GitHub para clonarlo en Apolo y compartirlo con el equipo.
- `data/`, `embeddings/`, pesos y la API key **nunca** se commitean.
- ML exige `model_development.ipynb` y el enlace al repo.

### 10.2 Equipo y roles

Equipo: **Samuel Llano Madrigal, Laura Andrea Castrillon y Jerónimo Campuzano**. Apoyo
académico: área de Microbiología y Parasitología de la UPB.

| Quién | Responsable principal **[Pendiente: confirmar]** |
|---|---|
| Jerónimo | Pipeline de datos, rama ML (embeddings + ML clásico + SHAP), app offline |
| Por asignar | Detector YOLO y entrenamiento en Apolo |
| Por asignar | Rama DL (CNN, VGG16, red liviana, ViT, Grad-CAM) |
| Todos | EDA, evaluación conjunta, contacto con la UPB, informes y presentaciones |

Regla de la sustentación de DL: **cualquier integrante debe poder responder sobre
cualquier componente**.

---

## 11. Entregables por materia

### Deep Learning

| Entregable | Fecha | Contenido |
|---|---|---|
| Correo de objetivos | **jue 29-oct-2026** | §2.1 |
| Informe (PDF, ≤ 3 págs) + presentación | **mié 11-nov-2026** (sin prórroga ni versiones posteriores) | Objetivos, base de datos, modelo, resultados, conclusiones |
| Sustentación | semanas 17–18 | 15 min + 5 de preguntas; participan todos. Sustentación 15 %, informe 5 % |

La guía de DL **no exige despliegue**: el camino crítico al 11-nov es YOLO + rama DL +
comparación + muestras externas.

### Machine Learning

| Entregable | Fecha | Contenido |
|---|---|---|
| Entrega completa | **3-nov-2026** | Notebook, repo y video ([checklist](entregas/machine-learning/README.md)) |
| Presentación | **19-nov-2026** | 50 % de la nota |
| Video demo | — | 2 min, modelo desplegado + interfaz |
| Repositorio + `model_development.ipynb` | — | Paso a paso, EDA profundo, justificación de decisiones |
| Presentación final | — | 50 % de la nota: problema, metodología, selección de modelo, generalización, métricas, despliegue, reajuste |

**[Propuesta]** Informar a ambos profesores que es un proyecto compartido y qué
componente corresponde a cada curso.

---

## 12. Cronograma **[Propuesta]**

| Fecha | Hito |
|---|---|
| 27–28 oct | Crear el repo en GitHub y clonarlo en Apolo; regenerar datos, recortes y embeddings con la fuente nueva (§5.2). |
| **29 oct** | Enviar correo de objetivos (DL). |
| 30 oct – 2 nov | YOLO en Apolo; rama ML: EDA, PCA, clustering, primeros clasificadores; línea base y detector de atajos. |
| **3 nov** | Entrega ML (por confirmar qué). |
| 3–7 nov | Rama DL (CNN base, VGG16, liviana, ViT) en Apolo; comparación de ramas; Grad-CAM; fotos externas y sus cajas. |
| 7–10 nov | Robustez y evaluación externa; informe y presentación de DL. |
| **11 nov** | Informe + presentación DL (límite duro). |
| 12–15 nov | App offline (ONNX); `model_development.ipynb`, SHAP, reajuste, video de 2 min. |
| 16–18 nov | **Jero en SC26** (competencia, remoto) → lo de ML listo el 15. |
| **19 nov** | Entrega final ML. |

Extras (§3) solo si el núcleo está listo. Strongyloides queda para después del semestre.

---

## 13. Riesgos

| Riesgo | Efecto | Mitigación |
|---|---|---|
| Brecha de dominio (dataset vs fotos de celular) | Errores en campo | Aumentaciones §5.8; prueba externa; monitoreo de *drift* |
| Atajo por fuente (la fuente o la resolución predicen la clase) | Precisión inflada | Detector de atajos, métricas por fuente, igualar nitidez, prueba externa |
| Fuga entre splits o entre detector y clasificador | Resultados optimistas | Partición por foto original, misma fuente y mismo split en todo, test congelado |
| Deformación ~7 % por "Stretch to 640" (§5.2) | Forma de los huevos levemente alterada | Reportarlo; *fork* sin redimensionar si se quiere eliminar |
| El modelo etiqueta cualquier cosa (Chula no tiene negativos) | Falsos positivos | Negativos externos, clase "otro", rechazo por confianza |
| Fotos de celular de mala calidad | Errores | Aviso de calidad, aumentos, protocolo de captura |
| Dataset pequeño y "demasiado fácil" | Métricas poco informativas | CV agrupada, curvas de aprendizaje, el valor está en lo externo |
| Uso como diagnóstico | Daño a pacientes | Siempre apoyo, confirmación experta, tarjeta del modelo |
| Tiempo: DL 11-nov sin prórroga; SC26 16–18 nov | Entregas incompletas | Cronograma §12; extras al final |
| Doble presentación del mismo proyecto | Problema académico | Avisar a ambos profesores |

Dependencias externas: **UPB** (imágenes externas, negativos, Strongyloides, revisión de
Grad-CAM), **profesores** (confirmar requisitos), **comité de ética** (solo si se usan
imágenes de pacientes).

---

## 14. Registro de decisiones

| Fecha | Decisión | Estado |
|---|---|---|
| 2026-10-23 | Rama ML = embeddings de imágenes + ML clásico (sugerencia del profesor de ML). | Decidido |
| 2026-10-23 | Extractor DINOv2 vía `torch.hub`; recorte rellenado a cuadrado. | Decidido |
| 2026-10-23 | Clasificar **recortes por huevo**, no fotos completas. | Decidido |
| 2026-10-23 | Copia parcial de Roboflow v4 YOLOv8 como primera fuente (v3/v7 cortan huevos). | Reemplazado |
| 2026-10-27 | Fuente: Chula completo `parasites-onxa6/parasites-trgfz` v1, YOLOv8. | Decidido |
| 2026-10-27 | Estructura del repo (`docs/entregas` por materia, `src/` por etapa, `data/` y `artifacts/` fuera de git). | Decidido |
| 2026-10-27 | Split propio 70/15/15 por foto, seed 42, compartido por todo el proyecto. | Decidido |
| 2026-10-27 | Salida en YOLO y COCO desde el mismo filtrado. | Decidido |
| 2026-10-27 | Arquitectura en dos etapas: YOLO detecta → rama ML y rama DL clasifican. | Decidido |
| 2026-10-27 | Entrenamiento en Apolo III (H100), no en Colab; PyTorch. | Decidido |
| 2026-10-27 | Entrega principal: laptop offline; celular como extra. | Decidido |
| 2026-10-27 | Fotos con celular por el ocular; conteo asistido por lámina (HPG) como estimación. | Decidido |
| 2026-10-27 | Rama DL: CNN base, VGG16, red liviana y ViT; receta de la ficha como punto de partida. | Decidido |
| 2026-10-27 | Clase "otro / sin parásito", Grad-CAM, detector de atajos, criterios de éxito (de la ficha). | Incorporado |
| 2026-10-27 | Corregir duplicados por reflejo con la opción A. | Descartado (la fuente nueva no tiene reflejos) |
| 2026-10-27 | DINOv2-S vs B; red liviana EfficientNet-B0 / MobileNetV3. | Propuesta |
| 2026-11-01 | Detector: YOLO11n de **una clase** (`item` = huevo) con aumentos de color y de celular (`h_single_aug`); umbral 0,25 (`compare_detectors.ipynb`). | Decidido |
| 2026-11-02 | Apolo sin VPN: la rama ML (recortes del detector, embeddings, modelos, app) se corre **en local**; los pesos de YOLO se trajeron de Apolo. Resultados idénticos a los de Apolo en `valid`. | Decidido |
| 2026-11-02 | Clasificador de la rama ML: **SVM lineal** (`C` = 0,003, 768 dimensiones de DINOv2 ViT-B/14 `cls`), elegido entre 6 familias por CV agrupada (`model_development.ipynb` §5). | Decidido |
| 2026-11-02 | Rechazo "consulte a un experto": confianza < 0,68 o similitud con el entrenamiento < 0,71 (percentil 1 de los huevos de `valid`). Los dudosos no se suman solos al conteo. | Decidido |
| 2026-11-02 | App offline en **Gradio** (`src/app/app.py`); registro local sin fotos para el monitoreo de *drift* (§9). | Decidido |
| 2026-11-02 | Entrega ML del 3-nov = todo (notebook, repo, video); presentación el 19-nov. | Decidido |

---

## 15. Diferencias con la ficha técnica y cómo se resolvieron

| Tema | Ficha técnica | Este plan (prevalece) |
|---|---|---|
| Arquitectura | Clasificador primero; YOLO en etapa 6; dos pasos "si compensa" | Dos etapas desde el inicio con dos ramas de clasificación |
| Entorno | Google Colab | Apolo III (H100) |
| Framework | PyTorch o TensorFlow | PyTorch |
| Despliegue | Celular de gama media | Laptop offline principal; celular extra |
| Gravedad de la infección | Fuera de alcance | Dentro: conteo por lámina y HPG como estimación de apoyo |
| ML clásico | No aparece | Rama ML con embeddings DINOv2 |
| Fuente de datos | Dataset v5 (recortes de Chula, ~900 por clase) | Mismo origen, pero con fotos completas y cajas: `parasites-trgfz` v1 (§5.2) |
| Segmentación SAM + U-Net | Opcional | Extra de baja prioridad |

Todo lo demás de la ficha (contexto, UPB, Strongyloides, requisitos, diccionario,
receta, evaluación, ética) se incorporó.

---

## 16. Preguntas abiertas

1. ¿Vale la pena hacer *fork* sin estiramiento para quitar la deformación del ~7 % (§5.2)?
2. Número de grupo (correo de DL) y reparto de roles entre los tres.
3. ~~Qué se entrega en ML el 3-nov y el 19-nov.~~ Resuelto: todo el 3-nov; presentación el 19-nov.
4. ¿El profesor de ML acepta una app local offline como "modelo desplegado" o exige un
   enlace público?
5. ¿La app es demo del curso o se busca uso real por una brigada? (Define cuánto se
   invierte en empaquetado y en implementar §9.)
6. Sistema operativo de la laptop de campo.
7. Qué preparación y aumento usan en el Chocó / la UPB (directo, lugol, Kato-Katz).
8. Cuándo llegan las imágenes externas de la UPB y las fotos de campo (§5.7).
9. ¿Ambos profesores aceptan el proyecto compartido?
10. Verificar citas: OMS (1.500 millones), estudio del Chocó, umbrales OMS, Chula ICIP 2022.

---

## 17. Glosario y referencias

**Glosario.** *Clasificación:* el modelo dice a qué clase pertenece una imagen.
*Detección (YOLO):* encuentra objetos en una foto, los encierra en una caja y los nombra.
*Segmentación:* dibuja el contorno exacto del objeto. *Transfer learning:* partir de una
red ya entrenada y ajustarla al problema. *Embedding:* vector que resume una imagen, sobre
el que se aplican modelos clásicos. *Vision Transformer:* red para imágenes basada en
atención. *Atajo:* el modelo acierta por una pista equivocada (estilo de la foto) y no por
la forma. *Fuga de datos:* imágenes iguales o casi iguales en entrenamiento y prueba.
*Grad-CAM:* mapa de calor de la zona que miró el modelo. *HPG:* huevos por gramo de heces.

**Referencias** (por completar con citas formales):

- Chula-ParasiteEgg-11 — ICIP 2022 Challenge *Parasitic Egg Detection and Classification
  in Microscopic Images*; copia completa en Roboflow (fuente usada, §5.2):
  https://universe.roboflow.com/parasites-onxa6/parasites-trgfz (v1, CC BY 4.0). Primera
  fuente, reemplazada (§5.4):
  https://universe.roboflow.com/research-t0os5/test-parasite-egg-detection.
- ParasitoBank (Universidad del Cauca); CDC DPDx; Atlas de Parasitología LACES (UFG).
- OMS — infecciones por geohelmintos; método Kato-Katz y umbrales de intensidad.
- Estudio de parasitosis intestinal en comunidades indígenas del Chocó (*Enferm. Infecc.
  Microbiol. Clin.*, 2023).
- Oquab et al., 2023 — *DINOv2*. Simonyan & Zisserman, 2015 — *VGG*. He et al., 2016 —
  *ResNet*. Tan & Le, 2019 — *EfficientNet*. Howard et al., 2019 — *MobileNetV3*.
  Dosovitskiy et al., 2021 — *ViT*. Selvaraju et al., 2017 — *Grad-CAM*. Lundberg & Lee,
  2017 — *SHAP*. Ultralytics YOLO; Redmon et al., 2016 — *YOLO*.
