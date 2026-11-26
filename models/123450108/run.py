
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

FOLDER = Path(__file__).resolve().parent
NIM = FOLDER.name
JALUR = "transformer"


def muat_model():
    berkas = FOLDER / "model.py"
    spec = importlib.util.spec_from_file_location("model_transformer", berkas)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")
    args = parser.parse_args()

    latih.atur_seed(args.seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    model_module = muat_model()
    kodek = getattr(data, "KODEK", None)
    keluaran = kodek.n_keluaran if kodek else int(ytr.max()) + 1
    dekode = kodek.dekode if kodek else lambda z: z.argmax(axis=1)

    mulai = time.time()
    model, info = model_module.jalankan(
        Xtr, ytr, Xva, yva, Xte,
        seed=args.seed,
        latih=latih,
        cetak="[Transformer]",
        keluaran=keluaran,
        kodek=kodek,
    )
    durasi = time.time() - mulai

    logit_val = latih.prediksi(model, Xva)
    pred_val = dekode(logit_val)

    pred_test = None
    if args.final and Xte is not None:
        pred_test = dekode(latih.prediksi(model, Xte))

    tambahan = {
        "hiperparameter": model_module.HIPER,
        "jumlah_parameter": latih.jumlah_parameter(model),
        "latensi_ms": latih.latensi_ms(model, Xva),
        **info,
    }

    latih.tulis_hasil(
        folder=NIM,
        jalur=JALUR,
        seed=args.seed,
        pred_val=pred_val,
        pred_test=pred_test,
        durasi=durasi,
        perangkat=args.perangkat,
        tambahan=tambahan,
    )


if __name__ == "__main__":
    main()
