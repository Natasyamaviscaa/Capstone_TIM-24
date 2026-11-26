"""Jalur CNN 1-D dengan opsi eksperimen ablation mean removal."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data
from src.tim import latih

FOLDER = Path(__file__).resolve().parent.name
JALUR = "cnn"


def muat_model():
    spec = importlib.util.spec_from_file_location(
        "model_jalur",
        Path(__file__).with_name("model.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hapus_mean(X):
    """Menghilangkan rata-rata setiap kanal pada setiap jendela."""
    return X - np.mean(X, axis=2, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--final",
        action="store_true",
        help="Evaluasi data uji sekali pada sprint terakhir"
    )
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument(
        "--ablation-riak",
        action="store_true",
        help="Eksperimen CNN dengan penghilangan mean per jendela"
    )
    a = ap.parse_args()

    # Pisahkan hasil baseline dan hasil ablation.
    folder_hasil = (
        f"{FOLDER}-ablation-riak"
        if a.ablation_riak
        else FOLDER
    )

    latih.atur_seed(a.seed)
    t0 = time.time()

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    if a.ablation_riak:
        print("[Ablation] Menghilangkan mean setiap jendela...")
        Xtr = hapus_mean(Xtr)
        Xva = hapus_mean(Xva)
        Xte = hapus_mean(Xte)

    kodek = getattr(data, "KODEK", None)
    keluaran = kodek.n_keluaran if kodek else int(ytr.max()) + 1
    dekode = kodek.dekode if kodek else (lambda z: z.argmax(1))

    jalur = muat_model()

    model, info = jalur.jalankan(
        Xtr, ytr, Xva, yva, Xte,
        seed=a.seed,
        latih=latih,
        cetak=folder_hasil,
        keluaran=keluaran,
        kodek=kodek
    )

    logit_val = latih.prediksi(model, Xva)
    pred_val = dekode(logit_val)

    # Data uji hanya diprediksi jika opsi --final digunakan.
    pred_test = (
        dekode(latih.prediksi(model, Xte))
        if a.final else None
    )

    durasi = time.time() - t0

    tambahan = {
        "jalur": JALUR,
        "seed": a.seed,
        "ablation_riak": a.ablation_riak,
        "hiper": jalur.HIPER,
        "parameter": latih.jumlah_parameter(model),
        "latensi_ms": latih.latensi_ms(model, Xva),
        **data.metrik_tim(yva, pred_val, logit=logit_val),
        **info
    }

    latih.tulis_hasil(
        folder_hasil,
        JALUR,
        a.seed,
        pred_val,
        pred_test,
        durasi,
        a.perangkat,
        tambahan
    )

    folder_result = AKAR / "results" / folder_hasil
    folder_result.mkdir(parents=True, exist_ok=True)

    np.save(folder_result / "logit_val.npy", logit_val)

    print(f"\nHasil eksperimen tersimpan di results/{folder_hasil}/")


if __name__ == "__main__":
    main()
