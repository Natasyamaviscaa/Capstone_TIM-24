"""Jalur RNN: GRU dua lapis atas deret sensor yang diperkecil.

Deret [kanal, T] dirata-rata per 8 sampel (400 Hz → 50 Hz, 50 langkah per detik) agar GRU tidak harus
mengingat ratusan langkah, lalu GRU 2 lapis (64 unit). Rata-rata keadaan tersembunyi atas waktu → linear.
"""
from __future__ import annotations

HIPER = {"epoch": 25, "lr": 2e-3, "batch": 64, "wd": 1e-4, "unit": 64, "lapis": 2, "gabung": 8, "dropout": 0.2}


def buat(n_kelas: int, n_kanal: int):
    import torch.nn as nn

    h = HIPER

    class Rnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_kanal, h["unit"], h["lapis"], batch_first=True, dropout=h["dropout"])
            self.kepala = nn.Sequential(nn.Dropout(h["dropout"]), nn.Linear(h["unit"], n_kelas))

        def forward(self, x):                                   # x [B, kanal, T]
            x = nn.functional.avg_pool1d(x, h["gabung"]).transpose(1, 2)
            out, _ = self.gru(x)
            return self.kepala(out.mean(1))

    return Rnn()


def jalankan(Xtr, ytr, Xva, yva, Xte, *, seed, latih, cetak="", keluaran=None, kodek=None):
    model = buat(keluaran or int(ytr.max()) + 1, Xtr.shape[1])
    info = latih.latih(model, Xtr, ytr, Xva, yva, seed=seed, epoch=HIPER["epoch"], lr=HIPER["lr"],
                       batch=HIPER["batch"], wd=HIPER["wd"], cetak=cetak, kodek=kodek)
    return model, info
