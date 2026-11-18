"""Jalur AE: autoencoder konvolusi 1-D, lalu klasifikasi dari ruang laten (encoder dibekukan).

Tahap 1 (tanpa label): AE merekonstruksi deret sensor data latih (MSE). Tahap 2: encoder beku (BatchNorm
mode evaluasi), laten dirata-rata atas waktu, MLP kecil dilatih dengan label.
"""
from __future__ import annotations

HIPER = {"epoch_ae": 25, "lr_ae": 2e-3, "epoch": 30, "lr": 2e-3, "batch": 64, "wd": 1e-4, "laten": 32}


def _ae(n_kanal: int):
    import torch.nn as nn

    k = HIPER["laten"]
    enc = nn.Sequential(nn.Conv1d(n_kanal, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.Conv1d(32, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.Conv1d(64, k, 7, 2, 3))
    dek = nn.Sequential(nn.ConvTranspose1d(k, 64, 8, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
                        nn.ConvTranspose1d(64, 32, 8, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
                        nn.ConvTranspose1d(32, n_kanal, 8, 2, 3))
    return enc, dek


def _latih_ae(enc, dek, Xtr, Xva, seed, cetak):
    import torch

    g = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(Xtr); V = torch.from_numpy(Xva)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dek.parameters()), lr=HIPER["lr_ae"], weight_decay=HIPER["wd"])

    def rekon(x):
        return dek(enc(x))[..., : x.shape[-1]]

    riwayat = []
    for e in range(1, HIPER["epoch_ae"] + 1):
        enc.train(); dek.train()
        for j in torch.randperm(len(X), generator=g).split(HIPER["batch"]):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(rekon(X[j]), X[j])
            loss.backward(); opt.step()
        enc.eval(); dek.eval()
        with torch.no_grad():
            mse = float(sum(torch.nn.functional.mse_loss(rekon(v), v, reduction="sum") for v in V.split(256)) / V.numel())
        riwayat.append({"epoch": e, "val_mse": round(mse, 4)})
        if cetak:
            print(f"[{cetak}/ae] epoch {e:2d} val_mse {mse:.4f}", flush=True)
    return riwayat


def jalankan(Xtr, ytr, Xva, yva, Xte, *, seed, latih, cetak="", keluaran=None, kodek=None):
    import torch.nn as nn

    enc, dek = _ae(Xtr.shape[1])
    riwayat_ae = _latih_ae(enc, dek, Xtr, Xva, seed, cetak)
    for p in enc.parameters():
        p.requires_grad = False
    inti = nn.Sequential(enc, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(HIPER["laten"], 64), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(64, keluaran or int(ytr.max()) + 1))

    class BekuBN(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def train(self, mode=True):
            super().train(mode)
            self.m[0].eval()
            return self

        def forward(self, x):
            return self.m(x)

    model = BekuBN(inti)
    info = latih.latih(model, Xtr, ytr, Xva, yva, seed=seed, epoch=HIPER["epoch"], lr=HIPER["lr"],
                       batch=HIPER["batch"], wd=HIPER["wd"], cetak=cetak, kodek=kodek)
    info["riwayat_ae"] = riwayat_ae
    return model, info
