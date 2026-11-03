# Guion del video demo (2 minutos)

Requisito: mostrar el modelo desplegado y su interfaz. Grabar la pantalla de la laptop con la app corriendo
**con el Wi-Fi apagado** (así se ve que funciona sin internet).

**Antes de grabar**

```bash
python src/app/app.py      # abrir http://127.0.0.1:7860 en el navegador, pantalla completa
```

Tener a mano tres fotos: una de cada especie (los ejemplos de la app sirven) y
`data/raw/valid/images/Ascaris-lumbricoides_0865_jpg.rf.f5323de6aec3cd7c04af9465bd201f68.jpg` (tiene un huevo dudoso, cortado por el borde).

| Tiempo | Pantalla | Qué decir |
|---|---|---|
| 0:00–0:15 | Título de la app | "En el Chocó las parasitosis por *Ascaris*, *Trichuris* y uncinarias son muy frecuentes, y contar huevos al microscopio exige un experto. Esta app apoya al microscopista: detecta, identifica y cuenta los huevos en fotos tomadas con el celular, sin internet." |
| 0:15–0:25 | Wi-Fi apagado | "Todo corre en esta laptop: no hay conexión." |
| 0:25–0:50 | Subir foto de *Trichuris* → Analizar | "Subo la foto del campo. Un detector YOLO encuentra el huevo, DINOv2 lo convierte en un vector y una SVM lineal decide la especie: *Trichuris*, muy segura. Tarda menos de un décimo de segundo." |
| 0:50–1:05 | Sumar a la lámina; repetir con *Ascaris* y uncinaria | "Cada foto se suma a la lámina en curso." |
| 1:05–1:30 | Foto `Ascaris-lumbricoides_0865` | "Aquí hay dos objetos. Uno es un *Ascaris* seguro; el otro sale en rojo: el modelo avisa que no se parece a los huevos conocidos, porque está cortado por el borde. No lo cuenta solo: pide consultar a un experto." |
| 1:30–1:50 | Tabla de la lámina | "Al final, el conteo por especie, el HPG estimado (conteo × 24, Kato-Katz) y la intensidad según la OMS. Es una estimación de apoyo, nunca un diagnóstico." |
| 1:50–2:00 | Cierre | "En la prueba final el clasificador tuvo F1 macro de 0,998, y el sistema contó exacto el 97 % de las fotos. El repositorio tiene el notebook con todo el desarrollo." |

**Cifras que se pueden decir** (todas salen de `model_development.ipynb`): F1 macro 0,990 en validación cruzada
y en `valid`, 0,998 en `test`; conteo exacto en el 96,6 % de 409 fotos de prueba; ~90 ms por foto en CPU.
