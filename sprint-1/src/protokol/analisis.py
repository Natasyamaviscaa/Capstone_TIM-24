"""Analisis kesalahan validasi → reports/kesalahan.md + reports/kesalahan.json

    python -m src.protokol.analisis

Per model: matriks kebingungan keadaan selang dan F1-makro per daya pompa (kartu kasus: jangan hanya gabungan).
"""
from __future__ import annotations

import json

import numpy as np

from src.common import manifest
from src.protokol import data

NAMA = {"baseline": "baseline starter", "contoh-cnn": "CNN", "contoh-rnn": "RNN", "contoh-trf": "Transformer",
        "contoh-ae": "AE", "non-dl": "non-DL"}
SINGKAT = ["lurus", "tertekuk", "dinaikkan"]


def main() -> None:
    akar = manifest.AKAR
    b = data.bagian(); c = b["c"]
    y = c["y"][b["val"]]; pompa = c["pompa"][b["val"]]
    hasil, rinci, kolom = {}, [], []
    for d in sorted((akar / "results").iterdir()):
        if not (d / "pred_val.npy").exists():
            continue
        p = np.load(d / "pred_val.npy").astype(int)
        nama = NAMA.get(d.name, d.name); kolom.append(nama)
        M = np.zeros((3, 3), int)
        for a, t in zip(y, p):
            M[a, t] += 1
        hasil[d.name] = {"matriks": M.tolist(), **data.metrik_tim(y, p),
                         "f1_per_daya": {int(k): data.metrik_tim(y[pompa == k], p[pompa == k])["val_f1_makro"]
                                         for k in sorted(set(pompa.tolist()))}}
        rinci += [f"### {nama}", "", "| benar \\ tebakan | " + " | ".join(SINGKAT) + " |", "|---|---|---|---|",
                  *[f"| {SINGKAT[i]} | " + " | ".join(str(x) for x in M[i]) + " |" for i in range(3)], ""]
    baris = ["# Analisis kesalahan (validasi, seed 1)", "",
             f"Validasi = ulangan {b['ulangan_val']} di blok latih ({len(y)} jendela). Validasi berada DI DALAM blok latih: "
             "lihat sidik blok di mutu-data.", "", "| | " + " | ".join(kolom) + " |", "|---|" + "---|" * len(kolom),
             "| **F1-makro** | " + " | ".join(f"**{hasil[d]['val_f1_makro']:.2f}**" for d in hasil) + " |"]
    for k in data.KELAS:
        baris.append(f"| F1 {k} | " + " | ".join(f"{hasil[d]['val_f1_rinci'][k]:.2f}" for d in hasil) + " |")
    for pw in sorted({k for d in hasil for k in hasil[d]["f1_per_daya"]}):
        baris.append(f"| F1-makro · pompa {pw} | " + " | ".join(f"{hasil[d]['f1_per_daya'][pw]:.2f}" for d in hasil) + " |")
    baris += ["", *rinci]
    (akar / "reports" / "kesalahan.md").write_text("\n".join(baris), encoding="utf-8")
    (akar / "reports" / "kesalahan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"{len(hasil)} model → reports/kesalahan.md")


if __name__ == "__main__":
    main()
