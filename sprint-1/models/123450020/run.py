
"""Jalur CNN 1D anggota 123450020."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data
from src.tim import latih

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"


def muat_model():
    spec = importlib.util.spec_from_file_location(
        "model_jalur", Path(__file__).with_name("model.py")
    )
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    latih.atur_seed(a.seed)
    mulai = time.time()

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    print("Input train:", Xtr.shape)
    print("Input validasi:", Xva.shape)
    print("Jumlah kelas:", len(np.unique(ytr)))

    jalur = muat_model()
    kodek = getattr(data, "KODEK", None)
    keluaran = kodek.n_keluaran if kodek else int(ytr.max()) + 1

    model, info = jalur.jalankan(
        Xtr, ytr, Xva, yva, Xte,
        seed=a.seed,
        latih=latih,
        cetak=NIM,
        keluaran=keluaran,
        kodek=kodek
    )

    logit_val = latih.prediksi(model, Xva)
    pred_val = (
        kodek.dekode(logit_val)
        if kodek else logit_val.argmax(axis=1)
    )

    pred_test = None
    if a.final:
        logit_test = latih.prediksi(model, Xte)
        pred_test = (
            kodek.dekode(logit_test)
            if kodek else logit_test.argmax(axis=1)
        )

    durasi = time.time() - mulai

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", f"results/{NIM}/pred_val.npy",
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", f"results/{NIM}/pred_test.npy"]

    subprocess.run(perintah, check=True, cwd=AKAR)

    print("Selesai. Hasil tersimpan di results/" + NIM)


if __name__ == "__main__":
    main()
