# Correo de objetivos — borrador

- **Para:** correo del prof. Raúl Andrés Castañeda Quintero
- **Asunto:** Proyecto Final – Grupo #XX – Objetivos del proyecto  ← reemplazar XX
- **Fecha límite:** jueves 29 de octubre de 2026

---

Buen día, profesor.

Le compartimos el título y los objetivos de nuestro proyecto final.

**Integrantes:** Samuel Llano Madrigal, Laura Andrea Castrillon y Jerónimo Campuzano.

**Título:** Detección, clasificación y conteo de geohelmintos (*Ascaris*, *Trichuris* y
uncinarias) en imágenes de microscopio mediante Deep Learning, como apoyo al diagnóstico
en comunidades del Chocó.

**Objetivo general:** Desarrollar y evaluar un sistema basado en Deep Learning que, a
partir de fotografías de microscopio tomadas con celular, detecte, clasifique y cuente
huevos de *Ascaris lumbricoides*, *Trichuris trichiura* y uncinarias, que funcione sin
conexión a internet y que mantenga su desempeño con imágenes de fuentes distintas a las
de entrenamiento.

**Objetivos específicos:**

1. Construir un conjunto de datos reproducible y sin fuga a partir de Chula-ParasiteEgg-11
   (partición por foto original en entrenamiento, validación y prueba, aumento de datos y
   diccionario de datos) y entrenar un detector de huevos YOLO con transfer learning.
2. Entrenar y comparar clasificadores de huevos sobre los mismos recortes —una CNN base,
   VGG16 y una red liviana (EfficientNet-B0 / MobileNetV3) con transfer learning, y un
   Vision Transformer—, reportando F1 macro, matriz de confusión y desempeño en imágenes
   externas.
3. Evaluar la robustez e interpretabilidad del mejor modelo (perturbaciones de brillo y
   nitidez, Grad-CAM) e integrarlo en una aplicación offline que lleve el conteo por
   lámina y estime la intensidad de infección.

Quedamos atentos a sus comentarios.

Saludos,
