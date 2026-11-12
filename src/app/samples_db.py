"""Local SQLite database of stool samples ("muestras") for the offline app.

The WHO Kato-Katz protocol reads two slides per sample. The result per sample and species is
HPG (eggs per gram) = mean of the eggs counted in the slides read x 24, with the WHO intensity
class. Only the slides that have at least one photo are used, so a sample with only slide 1
is estimated from slide 1 alone (the app says so).

Privacy: no images and no personal data. A sample is identified only by an anonymous code
typed by the user; a photo is stored as a short hash plus the model's signals per egg.
"""

import csv
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

KATO_KATZ_FACTOR = 24  # 41.7 mg template: eggs per gram = eggs counted in the whole slide x 24
SLIDES = (1, 2)
# WHO intensity thresholds in eggs per gram (Montresor et al., 1998): upper bounds of light and moderate
WHO_THRESHOLDS = {"ascaris": (4_999, 49_999), "trichuris": (999, 9_999), "hookworm": (1_999, 3_999)}
STATUS_OK = "ok"  # same value as inference.STATUS_OK

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    community TEXT NOT NULL DEFAULT '',
    technique TEXT NOT NULL DEFAULT 'Kato-Katz',
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS slides (
    id INTEGER PRIMARY KEY,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    slide_number INTEGER NOT NULL CHECK (slide_number IN (1, 2)),
    created_at TEXT NOT NULL,
    UNIQUE (sample_id, slide_number)
);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    slide_id INTEGER NOT NULL REFERENCES slides(id) ON DELETE CASCADE,
    photo_hash TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    n_eggs INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS eggs (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    species TEXT NOT NULL,
    prob REAL,
    similarity REAL,
    status TEXT NOT NULL
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path):
    """Open the database (created on first use), commit on success, always close."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA)
        with con:  # transaction: commit or roll back
            yield con


def normalize_code(code):
    """Codes are compared without surrounding spaces and without case: ' cho-1 ' == 'CHO-1'."""
    return " ".join((code or "").split()).upper()


def who_intensity(species, epg):
    if epg == 0:
        return "Sin huevos contados"
    light, moderate = WHO_THRESHOLDS[species]
    return "Leve" if epg <= light else "Moderada" if epg <= moderate else "Severa"


def create_sample(db_path, code, community="", notes=""):
    """New sample; returns its normalized code. ValueError if the code is empty or already used."""
    code = normalize_code(code)
    if not code:
        raise ValueError("Escriba un código para la muestra.")
    try:
        with connect(db_path) as con:
            con.execute("INSERT INTO samples (code, created_at, community, notes) VALUES (?, ?, ?, ?)",
                        (code, now(), (community or "").strip(), (notes or "").strip()))
    except sqlite3.IntegrityError:
        raise ValueError(f"Ya existe una muestra con el código {code}. Elíjala en la lista.") from None
    return code


def sample_codes(db_path):
    """Codes of every sample, newest first."""
    with connect(db_path) as con:
        return [row[0] for row in con.execute("SELECT code FROM samples ORDER BY created_at DESC, id DESC")]


def _sample_id(con, code):
    row = con.execute("SELECT id FROM samples WHERE code = ?", (normalize_code(code),)).fetchone()
    if row is None:
        raise ValueError(f"No existe la muestra {code}.")
    return row[0]


def add_photo(db_path, code, slide_number, photo_hash, eggs):
    """Store an analyzed photo and its eggs in slide 1 or 2 of a sample; returns the photo id.

    ValueError if the same photo (same hash) was already added to any slide of this sample:
    one microscope field cannot be counted twice.
    """
    slide_number = int(slide_number)
    if slide_number not in SLIDES:
        raise ValueError("La lámina debe ser 1 o 2.")
    with connect(db_path) as con:
        sample_id = _sample_id(con, code)
        used = con.execute("SELECT sl.slide_number FROM photos p JOIN slides sl ON p.slide_id = sl.id "
                           "WHERE sl.sample_id = ? AND p.photo_hash = ?", (sample_id, photo_hash)).fetchone()
        if used:
            raise ValueError(f"Esta foto ya se sumó a la lámina {used[0]} de esta muestra.")
        con.execute("INSERT OR IGNORE INTO slides (sample_id, slide_number, created_at) VALUES (?, ?, ?)",
                    (sample_id, slide_number, now()))
        slide_id = con.execute("SELECT id FROM slides WHERE sample_id = ? AND slide_number = ?",
                               (sample_id, slide_number)).fetchone()[0]
        photo_id = con.execute("INSERT INTO photos (slide_id, photo_hash, analyzed_at, n_eggs) VALUES (?, ?, ?, ?)",
                               (slide_id, photo_hash, now(), len(eggs))).lastrowid
        con.executemany("INSERT INTO eggs (photo_id, species, prob, similarity, status) VALUES (?, ?, ?, ?, ?)",
                        [(photo_id, e["species"], e.get("prob"), e.get("similarity"), e["status"]) for e in eggs])
    return photo_id


def sample_result(db_path, code):
    """Per-sample result: photos per slide, slides used, and per species the eggs of each slide,
    their mean over the slides used, HPG = mean x 24 and the WHO class. Doubtful eggs are only
    counted in 'review'."""
    with connect(db_path) as con:
        sample_id = _sample_id(con, code)
        code, created_at, community, technique = con.execute(
            "SELECT code, created_at, community, technique FROM samples WHERE id = ?", (sample_id,)).fetchone()
        photos = dict(con.execute("SELECT sl.slide_number, COUNT(p.id) FROM slides sl "
                                  "LEFT JOIN photos p ON p.slide_id = sl.id WHERE sl.sample_id = ? "
                                  "GROUP BY sl.slide_number", (sample_id,)).fetchall())
        rows = con.execute("SELECT sl.slide_number, e.species, e.status, COUNT(*) FROM eggs e "
                           "JOIN photos p ON e.photo_id = p.id JOIN slides sl ON p.slide_id = sl.id "
                           "WHERE sl.sample_id = ? GROUP BY 1, 2, 3", (sample_id,)).fetchall()
    photos = {n: photos.get(n, 0) for n in SLIDES}
    used = [n for n in SLIDES if photos[n] > 0]
    counts = {sp: {n: 0 for n in SLIDES} for sp in WHO_THRESHOLDS}
    review = 0
    for slide_number, species, status, n in rows:
        if status == STATUS_OK and species in counts:
            counts[species][slide_number] += n
        else:
            review += n
    species = {}
    for sp, per_slide in counts.items():
        mean = sum(per_slide[n] for n in used) / len(used) if used else None
        hpg = round(mean * KATO_KATZ_FACTOR) if used else None
        species[sp] = {"slides": per_slide, "mean": mean, "hpg": hpg,
                       "intensity": who_intensity(sp, hpg) if used else "—"}
    return {"code": code, "created_at": created_at, "community": community, "technique": technique,
            "photos": photos, "slides_used": used, "species": species, "review": review}


def all_results(db_path):
    """sample_result of every sample, newest first."""
    return [sample_result(db_path, code) for code in sample_codes(db_path)]


def export_csv(db_path, out_path):
    """One row per sample with the eggs per slide, mean, HPG and WHO class of each species."""
    header = ["codigo", "fecha", "comunidad", "tecnica", "fotos_lamina1", "fotos_lamina2", "laminas_usadas"]
    for sp in WHO_THRESHOLDS:
        header += [f"{sp}_lamina1", f"{sp}_lamina2", f"{sp}_promedio", f"{sp}_hpg", f"{sp}_intensidad_oms"]
    header.append("huevos_por_revisar")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel opens accents right
        writer = csv.writer(f)
        writer.writerow(header)
        for r in all_results(db_path):
            row = [r["code"], r["created_at"], r["community"], r["technique"], r["photos"][1], r["photos"][2],
                   "+".join(map(str, r["slides_used"]))]
            for sp in WHO_THRESHOLDS:
                s = r["species"][sp]
                row += [s["slides"][1], s["slides"][2], "" if s["mean"] is None else s["mean"],
                        "" if s["hpg"] is None else s["hpg"], s["intensity"]]
            row.append(r["review"])
            writer.writerow(row)
    return out_path
