"""Offline app: photo of a microscope field -> eggs found, species, confidence. Photos are added
to slide 1 or 2 of a sample (anonymous code) kept in a local SQLite database; the sample gets an
estimate of eggs per gram (Kato-Katz: mean of the two slides x 24) and the WHO intensity class.
Decision support only: an expert confirms every result.

Usage (from the repo root):
    python src/app/app.py            # http://127.0.0.1:7860
    python src/app/app.py --lan      # also reachable from a phone on the laptop's hotspot
    python src/app/app.py --db /tmp/test.db --port 7862   # throwaway database (tests)

Nothing leaves the laptop. Photos are not stored: the database keeps a hash of each photo and
the species / signals of its eggs (artifacts/app_data/muestras.db), and a local log keeps the
model's signals (artifacts/app_logs/predictions.csv), used to monitor drift.
"""

import argparse
import csv
import hashlib
import os
import sys
import tempfile
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
import samples_db as db  # noqa: E402
from samples_db import KATO_KATZ_FACTOR, WHO_THRESHOLDS, who_intensity  # noqa: E402,F401

LOG_PATH = ROOT / "artifacts" / "app_logs" / "predictions.csv"
DEFAULT_DB = ROOT / "artifacts" / "app_data" / "muestras.db"
EXAMPLES_DIR = ROOT / "data" / "raw" / "valid" / "images"

SPECIES = {
    "ascaris": {"name": "Ascaris lumbricoides", "color": "#2a78d6"},
    "trichuris": {"name": "Trichuris trichiura", "color": "#1baf7a"},
    "hookworm": {"name": "Uncinarias", "color": "#eb6834"},
}
DOUBTFUL_COLOR = "#b3261e"
SLIDE_CHOICES = [("Lámina 1", 1), ("Lámina 2", 2)]

DISCLAIMER = (
    "**Herramienta de apoyo.** No reemplaza el diagnóstico: un profesional confirma cada resultado. "
    "Funciona sin internet y no guarda las fotos."
)
HOW_TO = """
1. En **Muestra**, cree una muestra con un **código anónimo** (o elija una existente) y marque **Lámina 1** o **Lámina 2**.
2. **Suba o tome la foto** del campo del microscopio y pulse **Analizar foto**.
3. Revise las cajas: el color indica la especie. Las cajas **rojas con "?"** son dudosas: mírelas al microscopio.
4. Pulse **Sumar a la lámina** para guardar los huevos de esta foto en la lámina marcada.
5. Repita con cada campo donde vea un huevo. Si una lámina no tiene huevos, sume una foto de ella igual (cuenta 0).
6. Lea el resultado de la muestra: HPG = promedio de las dos láminas × 24. Puede exportar el historial en CSV.
"""
CODE_WARNING = ("**Use solo un código anónimo, nunca el nombre del paciente** (por ejemplo `CHO-2026-001`). "
                "No se guardan fotos ni datos personales.")


def confidence_words(egg):
    if egg["reason"] == "unfamiliar":
        return "Dudoso: no se parece a los huevos conocidos (¿cortado, desenfocado u otro objeto?). Consulte a un experto"
    if egg["reason"] == "between_species":
        return "Dudoso: duda entre especies. Consulte a un experto"
    return "Muy segura" if egg["prob"] >= 0.95 else "Segura"


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


def number_es(value):
    """2.5 -> '2,5'; 3.0 -> '3'."""
    return f"{value:g}".replace(".", ",")


def sample_table(result):
    """Per species: eggs of slide 1 and 2, mean, HPG and WHO class of the selected sample."""
    columns = ["Especie", "Huevos lámina 1", "Huevos lámina 2", "Promedio", "HPG estimado",
               "Intensidad OMS (estimada)"]
    if result is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, info in SPECIES.items():
        s = result["species"][key]
        used = result["slides_used"]
        rows.append(dict(zip(columns, [
            info["name"],
            *[s["slides"][n] if n in used else "—" for n in db.SLIDES],
            number_es(s["mean"]) if used else "—",
            s["hpg"] if used else "—",
            s["intensity"],
        ])))
    return pd.DataFrame(rows, columns=columns)


def sample_summary(result):
    if result is None:
        return "Cree o elija una muestra para ver su resultado."
    photos, used = result["photos"], result["slides_used"]
    text = (f"**Muestra {result['code']}**" + (f" · {result['community']}" if result["community"] else "")
            + f" · Lámina 1: {photos[1]} foto(s) · Lámina 2: {photos[2]} foto(s).")
    if len(used) == 1:
        missing = 2 if used == [1] else 1
        text += f"\n\n**Falta la lámina {missing}: el HPG usa solo la lámina {used[0]}.**"
    elif not used:
        text += "\n\nTodavía no hay fotos sumadas a esta muestra."
    if result["review"]:
        text += (f"\n\n**{result['review']} huevo(s) dudoso(s) por revisar** al microscopio: no están en el conteo. "
                 "Si confirma su especie, súmelos a mano.")
    text += (f"\n\nHPG = promedio de huevos de las láminas leídas × {KATO_KATZ_FACTOR} (Kato-Katz de 41,7 mg). "
             "Es una **estimación de apoyo**; solo vale si se fotografiaron todos los huevos de cada lámina.")
    return text


def history_table(results):
    columns = ["Código", "Fecha", "Comunidad", "Láminas", "Fotos", "Huevos contados",
               *[f"HPG {info['name'].split()[0]}" for info in SPECIES.values()], "Por revisar"]
    rows = []
    for r in results:
        used = r["slides_used"]
        rows.append(dict(zip(columns, [
            r["code"], r["created_at"][:16].replace("T", " "), r["community"],
            " y ".join(map(str, used)) or "—", sum(r["photos"].values()),
            sum(s["slides"][n] for s in r["species"].values() for n in used),
            *[r["species"][key]["hpg"] if used else "—" for key in SPECIES],
            r["review"],
        ])))
    return pd.DataFrame(rows, columns=columns)


def new_state():
    return {"slide": uuid.uuid4().hex[:8], "pending": None, "pending_hash": None}


def photo_hash(image):
    return hashlib.sha1(image.tobytes()).hexdigest()[:12]


def log_photo(state, image, eggs, model_created):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    when = datetime.now().isoformat(timespec="seconds")
    image_hash = photo_hash(image)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "slide", "photo_hash", "n_eggs", "egg", "species", "prob", "similarity",
                             "det_score", "status", "model_created"])
        for number, egg in enumerate(eggs, start=1):
            writer.writerow([when, state["slide"], image_hash, len(eggs), number, egg["species"], egg["prob"],
                             egg["similarity"], egg["det_score"], egg["status"], model_created])
        if not eggs:
            writer.writerow([when, state["slide"], image_hash, 0, "", "", "", "", "", "", model_created])


def build_app(pipeline, db_path=DEFAULT_DB):
    model_created = pipeline.bundle.get("created", "")

    def result_of(code):
        return db.sample_result(db_path, code) if code else None

    def analyze(image, state):
        if image is None:
            return None, eggs_table([]), "Primero suba una foto.", state, gr.update(interactive=False)
        eggs = pipeline.analyze(image)
        log_photo(state, image, eggs, model_created)
        state["pending"], state["pending_hash"] = eggs, photo_hash(image)
        if not eggs:
            message = ("**No se detectaron huevos** en esta foto. Si usted ve uno al microscopio, tome otra foto "
                       "más enfocada y centrada en el huevo. Si la lámina no tiene huevos, puede pulsar "
                       "**Sumar a la lámina** para registrarla con 0.")
            return image, eggs_table([]), message, state, gr.update(interactive=True)
        sure = [e for e in eggs if e["status"] == STATUS_OK]
        doubtful = len(eggs) - len(sure)
        found = ", ".join(f"{sum(e['species'] == k for e in sure)} {info['name']}"
                          for k, info in SPECIES.items() if any(e["species"] == k for e in sure))
        message = f"**Huevos encontrados:** {len(eggs)}." + (f" Seguros: {found}." if sure else "")
        if doubtful:
            message += f" **{doubtful} dudoso(s): consulte a un experto** (caja roja con \"?\")."
        return draw_eggs(image, eggs), eggs_table(eggs), message, state, gr.update(interactive=True)

    def add_to_slide(state, code, slide_number):
        eggs = state.get("pending")
        result = result_of(code)
        keep = (sample_summary(result), sample_table(result), history_table(db.all_results(db_path)))
        if eggs is None:
            return *keep, "Primero analice una foto.", state, gr.update(interactive=False)
        if not code:
            return *keep, "**Primero cree o elija una muestra** (arriba, en *Muestra*).", state, gr.update()
        try:
            db.add_photo(db_path, code, slide_number, state["pending_hash"], eggs)
        except ValueError as error:
            return *keep, f"**{error}**", state, gr.update(interactive=False)
        state["pending"] = None  # the same photo cannot be added twice
        result = result_of(code)
        sure = sum(e["status"] == STATUS_OK for e in eggs)
        message = (f"Foto sumada a la **lámina {int(slide_number)}** de **{code}**: {sure} huevo(s) contado(s)"
                   + (f", {len(eggs) - sure} por revisar." if len(eggs) > sure else "."))
        return (sample_summary(result), sample_table(result), history_table(db.all_results(db_path)), message,
                state, gr.update(interactive=False))

    def create_sample(code, community):
        try:
            code = db.create_sample(db_path, code, community)
        except ValueError as error:
            return gr.update(), f"**{error}**", gr.update(), gr.update()
        return (gr.update(choices=db.sample_codes(db_path), value=code),
                f"Muestra **{code}** creada. Marque la lámina y sume las fotos.", "", "")

    def show_sample(code):
        result = result_of(code)
        return sample_summary(result), sample_table(result)

    def refresh():
        codes = db.sample_codes(db_path)
        return gr.update(choices=codes), history_table(db.all_results(db_path))

    def export():
        out = Path(tempfile.mkdtemp(prefix="muestras_")) / f"muestras_{datetime.now():%Y%m%d_%H%M}.csv"
        db.export_csv(db_path, out)
        return gr.update(value=str(out), visible=True)

    examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
    examples = [str(p) for key in ("Ascaris", "Hookworm", "Trichuris")
                for p in [q for q in examples if q.name.startswith(key)][:1]]

    with gr.Blocks(title="Conteo de geohelmintos") as demo:
        state = gr.State(new_state())
        gr.Markdown("# Conteo de huevos de geohelmintos\n*Ascaris*, *Trichuris* y uncinarias en fotos de microscopio")
        gr.Markdown(DISCLAIMER)
        with gr.Accordion("Cómo se usa", open=False):
            gr.Markdown(HOW_TO)

        gr.Markdown("## Muestra")
        gr.Markdown(CODE_WARNING)
        with gr.Row():
            with gr.Column():
                code_box = gr.Textbox(label="Código de la nueva muestra", placeholder="CHO-2026-001", max_lines=1)
                community_box = gr.Textbox(label="Brigada o comunidad (opcional)", max_lines=1)
                create_button = gr.Button("Crear muestra")
            with gr.Column():
                sample_dd = gr.Dropdown(choices=db.sample_codes(db_path), value=None, label="Muestra en uso",
                                        info="Elija una muestra ya creada")
                slide_radio = gr.Radio(choices=SLIDE_CHOICES, value=1, label="Lámina que está leyendo")
                status_text = gr.Markdown()

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

        gr.Markdown("## Resultado de la muestra")
        sample_text = gr.Markdown(sample_summary(None))
        sample_df = gr.Dataframe(value=sample_table(None), label="Conteo por especie y lámina", interactive=False)

        gr.Markdown("## Historial de muestras")
        history_df = gr.Dataframe(value=history_table(db.all_results(db_path)), label="Muestras guardadas",
                                  interactive=False)
        with gr.Row():
            export_button = gr.Button("Exportar CSV")
            export_file = gr.File(label="CSV del historial", visible=False, interactive=False)

        analyze_button.click(analyze, [photo, state], [result_image, result_table, result_text, state, add_button],
                             api_name="analyze")
        add_button.click(add_to_slide, [state, sample_dd, slide_radio],
                         [sample_text, sample_df, history_df, status_text, state, add_button], api_name="add_to_slide")
        create_button.click(create_sample, [code_box, community_box], [sample_dd, status_text, code_box, community_box],
                            api_name="create_sample")
        sample_dd.change(show_sample, [sample_dd], [sample_text, sample_df], api_name="show_sample")
        export_button.click(export, None, [export_file], api_name="export")
        demo.load(refresh, None, [sample_dd, history_df])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database of samples")
    ap.add_argument("--lan", action="store_true", help="listen on every interface (laptop hotspot)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    for path in (args.weights, args.bundle):
        if not path.is_file():
            raise SystemExit(f"missing {path}: see the README (detector weights and notebook bundle)")

    demo = build_app(EggPipeline(args.weights, args.bundle), args.db)
    demo.launch(server_name="0.0.0.0" if args.lan else "127.0.0.1", server_port=args.port, share=False,
                footer_links=[], theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
