"""Model Transformer Encoder untuk klasifikasi kondisi selang pompa.

Input: [batch, kanal, 200] — sinyal bus_mV selama 1 detik pada 200 Hz.
Setiap 20 sampel dijadikan satu token, sehingga ada 10 token per jendela.
"""
from __future__ import annotations

HIPER = {
    "epoch": 30,
    "lr": 1e-3,
    "batch": 64,
    "wd": 1e-2,
    "d": 64,
    "lapis": 2,
    "kepala": 4,
    "ffn": 128,
    "dropout": 0.1,
    "tambalan": 20,
    "maks_token": 256,
}


def buat(n_kelas: int, n_kanal: int):
    import torch
    import torch.nn as nn

    h = HIPER

    class Trf(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv1d(
                n_kanal, h["d"],
                kernel_size=h["tambalan"],
                stride=h["tambalan"],
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, h["d"]))
            self.pos = nn.Parameter(
                torch.zeros(1, h["maks_token"] + 1, h["d"])
            )
            nn.init.trunc_normal_(self.pos, std=0.02)

            lapis = nn.TransformerEncoderLayer(
                d_model=h["d"],
                nhead=h["kepala"],
                dim_feedforward=h["ffn"],
                dropout=h["dropout"],
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(
                lapis, h["lapis"], enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(h["d"])
            self.kepala = nn.Linear(h["d"], n_kelas)

        def forward(self, x):
            # x: [batch, kanal, 200]
            x = self.proj(x).transpose(1, 2)
            x = torch.cat(
                [self.cls.expand(len(x), -1, -1), x], dim=1
            )
            x = x + self.pos[:, :x.shape[1]]
            x = self.enc(x)
            return self.kepala(self.norm(x[:, 0]))

    return Trf()

def jalankan(Xtr, ytr, Xva, yva, Xte, *, seed, latih, cetak="", keluaran=None, kodek=None):
    model = buat(keluaran or int(ytr.max()) + 1, Xtr.shape[1])

    info = latih.latih(
        model,
        Xtr, ytr,
        Xva, yva,
        seed=seed,
        epoch=HIPER["epoch"],
        lr=HIPER["lr"],
        batch=HIPER["batch"],
        wd=HIPER["wd"],
        cetak=cetak,
        kodek=kodek,
    )

    return model, info