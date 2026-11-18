"""Jalur CNN: CNN 1-D atas deret sensor [kanal, T].

Tiga blok Conv1d (kernel 7) → BatchNorm → ReLU → MaxPool 2 (32 → 64 → 64 kanal), rata-rata global atas waktu,
dropout, linear. Kernel 7 pada 400 Hz ≈ 17 ms: cukup untuk pola getaran satu putaran bilah kipas.
"""
from __future__ import annotations

HIPER = {"epoch": 25, "lr": 2e-3, "batch": 64, "wd": 1e-4, "dropout": 0.2}


def buat(n_kelas: int, n_kanal: int):
    import torch.nn as nn

    def blok(a, b):
        return nn.Sequential(nn.Conv1d(a, b, 7, padding=3, bias=False), nn.BatchNorm1d(b), nn.ReLU(), nn.MaxPool1d(2))

    return nn.Sequential(blok(n_kanal, 32), blok(32, 64), blok(64, 64), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                         nn.Dropout(HIPER["dropout"]), nn.Linear(64, n_kelas))


def jalankan(Xtr, ytr, Xva, yva, Xte, *, seed, latih, cetak="", keluaran=None, kodek=None):
    model = buat(keluaran or int(ytr.max()) + 1, Xtr.shape[1])
    info = latih.latih(model, Xtr, ytr, Xva, yva, seed=seed, epoch=HIPER["epoch"], lr=HIPER["lr"],
                       batch=HIPER["batch"], wd=HIPER["wd"], cetak=cetak, kodek=kodek)
    return model, info
