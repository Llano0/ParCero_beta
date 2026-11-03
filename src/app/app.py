"""Offline app: photo of a microscope field -> eggs found, species, confidence, and the
running count of the slide with an estimate of eggs per gram (Kato-Katz) and the WHO
intensity class. Decision support only: an expert confirms every result.

Usage (from the repo root):
    python src/app/app.py            # http://127.0.0.1:7860
    python src/app/app.py --lan      # also reachable from a phone on the laptop's hotspot

Nothing leaves the laptop. Photos are not stored; only a local log with a hash of the
photo and the model's signals (artifacts/app_logs/predictions.csv), used to monitor drift.
"""

import argparse
import csv
import hashlib
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("YOLO_OFFLINE", "1")

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import DEFAULT_BUNDLE, DEFAULT_WEIGHTS, ROOT, STATUS_OK, EggPipeline  # noqa: E402

LOG_PATH = ROOT / "artifacts" / "app_logs" / "predictions.csv"
EXAMPLES_DIR = ROOT / "data" / "raw" / "valid" / "images"

KATO_KATZ_FACTOR = 24  # 41.7 mg template: eggs per gram = eggs counted in the whole slide x 24
SPECIES = {
    "ascaris": {"name": "Ascaris lumbricoides", "color": "#2a78d6"},
    "trichuris": {"name": "Trichuris trichiura", "color": "#1baf7a"},
    "hookworm": {"name": "Uncinarias", "color": "#eb6834"},
}
DOUBTFUL_COLOR = "#b3261e"
# WHO intensity thresholds in eggs per gram (Montresor et al., 1998): upper bounds of light and moderate
WHO_THRESHOLDS = {"ascaris": (4_999, 49_999), "trichuris": (999, 9_999), "hookworm": (1_999, 3_999)}

DISCLAIMER = (
    "**Herramienta de apoyo.** No reemplaza el diagnóstico: un profesional confirma cada resultado. "
    "Funciona sin internet y no guarda las fotos."
)
HOW_TO = """
1. **Suba o tome la foto** del campo del microscopio y pulse **Analizar foto**.
2. Revise las cajas: el color indica la especie. Las cajas **rojas con "?"** son dudosas: mírelas al microscopio.
3. Pulse **Sumar a la lámina** para agregar los huevos de esta foto al conteo.
4. Repita con cada campo donde vea un huevo. Al terminar la lámina, lea el resumen. **Nueva lámina** empieza de cero.
"""


def confidence_words(egg):
    if egg["reason"] == "unfamiliar":
        return "Dudoso: no se parece a los huevos conocidos (¿cortado, desenfocado u otro objeto?). Consulte a un experto"
    if egg["reason"] == "between_species":
        return "Dudoso: duda entre especies. Consulte a un experto"
    return "Muy segura" if egg["prob"] >= 0.95 else "Segura"


def who_intensity(species, epg):
    if epg == 0:
        return "Sin huevos contados"
    light, moderate = WHO_THRESHOLDS[species]
    return "Leve" if epg <= light else "Moderada" if epg <= moderate else "Severa"


def draw_eggs(image, eggs):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width = max(2, round(max(image.size) / 200))
    font = ImageFont.load_default(size=max(14, round(max(image.size) / 35)))
    for number, egg in enumerate(eggs, start=1):
        ok = egg["status"] == STATUS_OK
        color = SPECIES[egg["species"]]["color"] if ok else DOUBTFUL_COLOR
        x1, y1, x2, y2 = egg["box"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{number}. {SPECIES[egg['species']]['name'].split()[0]}" + ("" if ok else " ?")
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_h = text_box[3] - text_box[1] + 6
        top = y1 - text_h if y1 - text_h >= 0 else y2
        draw.rectangle((x1, top, x1 + text_box[2] - text_box[0] + 8, top + text_h), fill=color)
        draw.text((x1 + 4, top + 2), label, fill="white", font=font)
    return image


def eggs_table(eggs):
    return pd.DataFrame([{
        "N°": number,
        "Especie": SPECIES[egg["species"]]["name"],
        "Confianza": f"{egg['prob']:.0%}" if egg["status"] == STATUS_OK else "—",
        "Resultado": confidence_words(egg),
    } for number, egg in enumerate(eggs, start=1)], columns=["N°", "Especie", "Confianza", "Resultado"])


def slide_table(state):
    rows = []
    for key, info in SPECIES.items():
        count = state["counts"][key]
        epg = count * KATO_KATZ_FACTOR
        rows.append({"Especie": info["name"], "Huevos contados": count, "HPG estimado": epg,
                     "Intensidad OMS (estimada)": who_intensity(key, epg)})
    return pd.DataFrame(rows)


def slide_summary(state):
    review = state["review"]
    text = f"**Lámina en curso:** {state['photos']} foto(s) sumada(s)."
    if review:
        text += (f" **{review} huevo(s) dudoso(s) por revisar** al microscopio: no están en el conteo. "
                 "Si confirma su especie, súmelos a mano.")
    text += (f"\n\nHPG = huevos contados en toda la lámina × {KATO_KATZ_FACTOR} (Kato-Katz de 41,7 mg). "
             "Es una **estimación de apoyo**; solo vale si se fotografiaron todos los huevos de la lámina.")
    return text


def new_state():
    return {"slide": uuid.uuid4().hex[:8], "photos": 0, "counts": {k: 0 for k in SPECIES}, "review": 0,
            "pending": None}


def log_photo(state, image, eggs, model_created):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    photo_hash = hashlib.sha1(image.tobytes()).hexdigest()[:12]
    when = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "slide", "photo_hash", "n_eggs", "egg", "species", "prob", "similarity",
                             "det_score", "status", "model_created"])
        for number, egg in enumerate(eggs, start=1):
            writer.writerow([when, state["slide"], photo_hash, len(eggs), number, egg["species"], egg["prob"],
                             egg["similarity"], egg["det_score"], egg["status"], model_created])
        if not eggs:
            writer.writerow([when, state["slide"], photo_hash, 0, "", "", "", "", "", "", model_created])


def build_app(pipeline):
    model_created = pipeline.bundle.get("created", "")

    def analyze(image, state):
        if image is None:
            return None, eggs_table([]), "Primero suba una foto.", state, gr.update(interactive=False)
        eggs = pipeline.analyze(image)
        log_photo(state, image, eggs, model_created)
        state["pending"] = eggs
        if not eggs:
            message = ("**No se detectaron huevos** en esta foto. Si usted ve uno al microscopio, tome otra foto "
                       "más enfocada y centrada en el huevo.")
            return image, eggs_table([]), message, state, gr.update(interactive=False)
        sure = [e for e in eggs if e["status"] == STATUS_OK]
        doubtful = len(eggs) - len(sure)
        found = ", ".join(f"{sum(e['species'] == k for e in sure)} {info['name']}"
                          for k, info in SPECIES.items() if any(e["species"] == k for e in sure))
        message = f"**Huevos encontrados:** {len(eggs)}." + (f" Seguros: {found}." if sure else "")
        if doubtful:
            message += f" **{doubtful} dudoso(s): consulte a un experto** (caja roja con \"?\")."
        return draw_eggs(image, eggs), eggs_table(eggs), message, state, gr.update(interactive=True)

    def add_to_slide(state):
        eggs = state.get("pending")
        if not eggs:
            return slide_table(state), slide_summary(state), state, gr.update(interactive=False)
        for egg in eggs:
            if egg["status"] == STATUS_OK:
                state["counts"][egg["species"]] += 1
            else:
                state["review"] += 1
        state["photos"] += 1
        state["pending"] = None  # the same photo cannot be added twice
        return slide_table(state), slide_summary(state), state, gr.update(interactive=False)

    def reset_slide(state):
        state = new_state()
        return slide_table(state), slide_summary(state), state, gr.update(interactive=False)

    examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
    examples = [str(p) for key in ("Ascaris", "Hookworm", "Trichuris")
                for p in [q for q in examples if q.name.startswith(key)][:1]]

    with gr.Blocks(title="Conteo de geohelmintos") as demo:
        state = gr.State(new_state())
        gr.Markdown("# Conteo de huevos de geohelmintos\n*Ascaris*, *Trichuris* y uncinarias en fotos de microscopio")
        gr.Markdown(DISCLAIMER)
        with gr.Accordion("Cómo se usa", open=False):
            gr.Markdown(HOW_TO)
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Foto del campo del microscopio",
                                 sources=["upload", "clipboard"], height=420)
                analyze_button = gr.Button("Analizar foto", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=photo, label="Fotos de ejemplo (validación de Chula)")
            with gr.Column():
                result_image = gr.Image(type="pil", label="Resultado", interactive=False, height=420)
                result_text = gr.Markdown()
                result_table = gr.Dataframe(value=eggs_table([]), label="Huevos de esta foto", interactive=False)
                add_button = gr.Button("Sumar a la lámina", interactive=False)
        gr.Markdown("## Lámina")
        slide_text = gr.Markdown(slide_summary(new_state()))
        slide_df = gr.Dataframe(value=slide_table(new_state()), label="Conteo por especie", interactive=False)
        reset_button = gr.Button("Nueva lámina")

        analyze_button.click(analyze, [photo, state], [result_image, result_table, result_text, state, add_button])
        add_button.click(add_to_slide, [state], [slide_df, slide_text, state, add_button])
        reset_button.click(reset_slide, [state], [slide_df, slide_text, state, add_button])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--lan", action="store_true", help="listen on every interface (laptop hotspot)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    for path in (args.weights, args.bundle):
        if not path.is_file():
            raise SystemExit(f"missing {path}: see the README (detector weights and notebook bundle)")

    demo = build_app(EggPipeline(args.weights, args.bundle))
    demo.launch(server_name="0.0.0.0" if args.lan else "127.0.0.1", server_port=args.port, share=False,
                footer_links=[], theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
