"""Pembanding non-deep learning untuk deret sensor: fitur statistik + FFT + Random Forest / SVM.

    python models/non-dl/run.py --seed 1 [--final]

Per kanal (sebelum normalisasi): rata-rata, simpangan, RMS, puncak, kemencengan, keruncingan, dan energi
8 pita FFT dalam log (``src.protokol.deret.fitur_statistik``). Dua model dicoba pada data latih dan dipilih
dengan macro-F1 validasi; keduanya dicatat di tambahan.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data, deret  # noqa: E402
from src.tim import latih  # noqa: E402

FOLDER = Path(__file__).resolve().parent.name


def ringkas(X: np.ndarray) -> np.ndarray:
    return deret.fitur_statistik(X, data.LAJU)


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, _ = data.muat(standar=False)
    Ftr, Fva = ringkas(Xtr), ringkas(Xva)
    kodek = getattr(data, "KODEK", None)            # multi-label: satu kolom target per label (sklearn multi-output)
    Ytr = kodek.pisah(ytr) if kodek else ytr
    gabung = kodek.gabung if kodek else (lambda P: P)
    calon = {
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     random_state=a.seed, n_jobs=1),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", random_state=a.seed)),
    }
    if kodek:
        from sklearn.multioutput import MultiOutputClassifier
        calon["svm"] = MultiOutputClassifier(calon["svm"])
    skor = {}
    for nama, m in calon.items():
        m.fit(Ftr, Ytr)
        p = gabung(m.predict(Fva))
        f1 = kodek.skor(yva, p) if kodek else latih.macro_f1(yva, p)
        skor[nama] = {"val_f1": round(f1, 4), "val_acc": round(float((p == yva).mean()), 4), **data.metrik_tim(yva, p)}
    pilih = max(skor, key=lambda k: skor[k]["val_f1"])
    m = calon[pilih]
    pred_val = gabung(m.predict(Fva))
    pred_test = gabung(m.predict(ringkas(Xte))) if a.final else None
    durasi = time.time() - t0
    tambahan = {"jalur": "non-dl", "seed": a.seed, "dipilih": pilih, "calon": skor, "fitur": "statistik + energi 8 pita FFT per kanal",
                **data.metrik_tim(yva, pred_val)}
    latih.tulis_hasil(FOLDER, f"non-dl-{pilih}", a.seed, pred_val, pred_test, durasi, a.perangkat, tambahan)


if __name__ == "__main__":
    main()
