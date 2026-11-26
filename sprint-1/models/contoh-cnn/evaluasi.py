
import sys
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data
from src.tim import latih

FILE_MODEL = Path(__file__).with_name("model.py")
spec = importlib.util.spec_from_file_location("model_cnn", FILE_MODEL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Muat data
Xtr, ytr, Xva, yva, Xte, _ = data.muat()

# Latih ulang CNN baseline dengan seed 1
latih.atur_seed(1)

kodek = getattr(data, "KODEK", None)
keluaran = kodek.n_keluaran if kodek else int(ytr.max()) + 1

model, info = mod.jalankan(
    Xtr, ytr, Xva, yva, Xte,
    seed=1,
    latih=latih,
    keluaran=keluaran,
    kodek=kodek
)

# Prediksi validation
logit = latih.prediksi(model, Xva)

if kodek:
    pred = kodek.dekode(logit)
else:
    pred = logit.argmax(axis=1)

# Evaluasi
print("\nCLASSIFICATION REPORT")
print(classification_report(yva, pred, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(yva, pred))

print("\nJumlah data validasi:", len(yva))
