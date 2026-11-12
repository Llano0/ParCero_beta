# App de demostración

Aplicación local e ilustrativa del sistema. **Es una herramienta de apoyo, no un diagnóstico.**

Recorrido: se sube una foto del microscopio y YOLO de 1 clase (`dl/outputs/yolo/yolo_1class_aug`)
detecta los huevos. Cada huevo se recorta con 15 % de margen y EfficientNet-B0 (las 3 semillas,
promediadas, de `dl/outputs/transfer_efficientnet_b0`) da la especie y la confianza. Si la
confianza es menor a 65 %, el huevo queda marcado para revisión de un profesional.

La página lleva el registro de la muestra foto por foto: huevos por especie, HPG = total × 24
(Kato-Katz, 41,7 mg) e intensidad según la OMS (PLAN §6.4). Todo queda en el navegador: no hay base
de datos y no se guarda nada en disco. Se puede exportar el registro a CSV.

## Cómo correrla

Desde la raíz del repo, con el entorno `geohelmintos`. Solo usa librerías que ya están en el
entorno: el servidor es de la librería estándar de Python.

```bash
python app/app.py              # abre http://127.0.0.1:7860
python app/app.py --port 7861  # si el puerto está ocupado
```

- **En Apolo, desde VS Code (Remote-SSH):** al arrancar, VS Code ofrece reenviar el puerto. Si no
  lo hace, ve a la pestaña *Ports* y agrega el 7860. Luego abre `http://localhost:7860` en tu
  navegador. Corre en CPU, en el nodo de login, y tarda unos 1–2 s por foto.
- **En un portátil:** hacen falta los pesos, que no están en git. Copia
  `dl/outputs/yolo/yolo_1class_aug/weights/best.pt` y
  `dl/outputs/transfer_efficientnet_b0/seed{42,43,44}/best.pt` a las mismas rutas del repo local.

El servidor escucha solo en `127.0.0.1`, así que no queda expuesto en la red.

## Archivos

| Archivo | Qué hace |
|---|---|
| `app.py` | Carga los modelos y responde `POST /analyze`: foto → cajas, especie, confianza e imagen anotada |
| `static/index.html` | La interfaz: subir la foto, ver los huevos, guardarlos en la muestra y ver el resultado |

## Límites

- Es una demostración y no está validada para uso clínico.
- El HPG solo es una estimación válida si las fotos recorren toda la lámina sin repetir campos.
- La confianza no está calibrada.
- No hay clase "otro": cualquier objeto que YOLO marque se asigna a una de las 3 especies.
- La versión offline final del PLAN §8 (ONNX Runtime) sigue pendiente.
