"""Jalur TL (transfer learning): ResNet18 pralatih ImageNet diadaptasi ke sinyal meja 1-D.

Kasus Tim 24: klasifikasi keadaan selang pompa (3 kelas) dari sinyal listrik. Masukan protokol tim
``X [N, 1, 200]`` (kanal ``bus_mV``, jendela 1 s @ 200 Hz).

Karena pralatih ResNet18 adalah jaringan citra 2-D, tiap jendela 1-D dipetakan menjadi citra
"spektrogram": STFT (``n_fft=32``, ``hop=8``) -> magnitudo ``[kanal, frek, waktu]``, distandarkan per
(kanal, frek) memakai statistik data latih, lalu digandakan ke 3 kanal (R=G=B) agar cocok dengan
``conv1`` 3-kanal pralatih. Seluruh pemetaan ini berada DI DALAM ``forward`` sehingga model menerima
``X`` mentah protokol apa adanya.

Berkas ini **mandiri**: hanya butuh ``torch``/``numpy`` (tidak bergantung pada ``src.tim`` atau
``src.protokol``), sehingga bisa dijalankan di repo tim mana pun. ``run.py`` yang memanggil
``data.muat()`` dan menulis hasil lewat ``src.common.evaluate``.

Transfer learning dua tahap:
1. **Fitur beku** — badan ResNet18 (bobot ImageNet) dibekukan; hanya kepala linear baru (512 -> n_kelas)
   yang dilatih (``epoch_beku``).
2. **Fine-tune** — ``layer3`` & ``layer4`` + kepala dilatih dengan laju lebih kecil, memilih bobot
   dengan F1-makro validasi terbaik (``epoch``).

Bobot pralatih diunduh lewat ``torch.hub.load_state_dict_from_url`` (tanpa ``torchvision`` agar tidak
bentrok versi). Bila unduhan gagal (offline), arsitektur tetap berjalan dengan bobot acak dan
``status_pretrained`` menandainya di ``info`` — sprint tidak bergantung internet.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

# Batasi thread agar tidak oversubscribe (CPU laptop); bisa ditimpa lewat env TIM_THREADS.
torch.set_num_threads(int(os.environ.get("TIM_THREADS", "4")))

HIPER = {
    "epoch": 16,          # epoch tahap fine-tune
    "epoch_beku": 8,      # epoch tahap fitur beku
    "lr": 1e-3,           # laju tahap fitur beku (kepala)
    "lr_ft": 3e-4,        # laju tahap fine-tune
    "batch": 64,
    "wd": 1e-4,
    "n_fft": 32,
    "hop": 8,
    "lapis_buka": ("layer3", "layer4"),
    "pretrained": True,
}

_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


# ------------------------------------------------------------------------------------------------
# ResNet18 (identik torchvision) murni-torch agar bobot ImageNet termuat strict, + STFT di forward.
# ------------------------------------------------------------------------------------------------
def _conv3x3(i, o, s=1):
    return nn.Conv2d(i, o, 3, s, 1, bias=False)


def _conv1x1(i, o, s=1):
    return nn.Conv2d(i, o, 1, s, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(out + idt)


def _stft_mag(x, n_fft, hop, window):
    """x [B, C, T] -> magnitudo STFT [B, C, freq, frame] (diferensiabel)."""
    frame = x.unfold(-1, n_fft, hop) * window          # [B, C, frame, n_fft]
    z = torch.fft.rfft(frame, dim=-1)                   # [B, C, frame, freq]
    return z.abs().transpose(-1, -2)                    # [B, C, freq, frame]


class TLNet(nn.Module):
    """ResNet18 + front-end spektrogram; menerima [B, C, T] mentah -> logit [B, n_kelas]."""

    def __init__(self, n_kelas, n_kanal, n_fft, hop):
        super().__init__()
        self.n_kanal = n_kanal
        self.n_fft, self.hop = n_fft, hop
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)   # 3 kanal: spektrogram digandakan R=G=B
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_kelas)
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mu", torch.zeros(1, n_kanal, n_fft // 2 + 1, 1))
        self.register_buffer("sd", torch.ones(1, n_kanal, n_fft // 2 + 1, 1))
        self.pretrained_status = "acak"

    def _layer(self, planes, blok, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(_conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        lapis = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blok):
            lapis.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*lapis)

    def set_norm(self, mu, sd):
        self.mu = mu.detach().clone()
        self.sd = sd.detach().clone()

    def forward(self, x):
        if x.dim() == 4:                       # sudah citra [B, C, freq, frame]
            img = x
        else:                                  # [B, C, T] mentah
            if x.dim() == 2:
                x = x.unsqueeze(1)
            S = _stft_mag(x, self.n_fft, self.hop, self.window)
            img = (S - self.mu) / (self.sd + 1e-6)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
        h = self.maxpool(self.relu(self.bn1(self.conv1(img))))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return self.fc(self.avgpool(h).flatten(1))


def buat(n_kelas: int, n_kanal: int):
    """Bangun TLNet; muat bobot ImageNet ResNet18 (strict) bila tersedia."""
    model = TLNet(n_kelas, n_kanal, HIPER["n_fft"], HIPER["hop"])
    if HIPER.get("pretrained", True):
        try:
            sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu", check_hash=True)
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            _missing, unexpected = model.load_state_dict(sd, strict=False)
            model.pretrained_status = "imagenet" if not unexpected else f"imagenet(+{unexpected[:3]})"
        except Exception as e:            # offline / hash gagal: bobot acak
            model.pretrained_status = f"gagal:{type(e).__name__}"
    return model


# ------------------------------------------------------------------------------------------------
# Pembantu latih/prediksi (mandiri, tanpa src.tim)
# ------------------------------------------------------------------------------------------------
def _macro_f1(y, p) -> float:
    f = []
    for c in np.unique(np.concatenate([np.asarray(y), np.asarray(p)])):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def prediksi(model, X: np.ndarray, batch: int = 128) -> np.ndarray:
    """Logit [N, K] dalam mode evaluasi."""
    model.eval()
    keluar = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            keluar.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))).numpy())
    return np.concatenate(keluar)


def _statistik_norm(model, Xtr):
    """Hitung mu/sd spektrogram dari data latih (tanpa gradien)."""
    model.eval()
    akum, n = None, 0
    with torch.no_grad():
        for i in range(0, len(Xtr), 128):
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[i:i + 128]))
            if xb.dim() == 2:
                xb = xb.unsqueeze(1)
            S = _stft_mag(xb, model.n_fft, model.hop, model.window)
            s = S.sum(dim=(0, 3), keepdim=True)
            ss = (S ** 2).sum(dim=(0, 3), keepdim=True)
            akum = (s, ss) if akum is None else (akum[0] + s, akum[1] + ss)
            n += S.shape[0] * S.shape[3]
    mu = akum[0] / n
    sd = (akum[1] / n - mu ** 2).clamp_min(1e-8).sqrt()
    model.set_norm(mu, sd)


def _latih_tahap(model, Xtr, ytr, Xva, yva, *, seed, epoch, lr, wd, batch, cetak, tag):
    """AdamW + cosine; simpan bobot dengan F1-makro validasi terbaik. Kembalikan riwayat."""
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr))
    yt = torch.from_numpy(np.asarray(ytr, np.int64))
    riwayat, terbaik, bobot = [], (-1.0, -1), None
    for e in range(1, epoch + 1):
        model.train()
        total = 0.0
        for j in torch.randperm(len(Xt), generator=g).split(batch):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xt[j]), yt[j])
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        sched.step()
        p = prediksi(model, Xva).argmax(1)
        f1 = _macro_f1(yva, p)
        riwayat.append({"epoch": e, "loss": round(total / len(Xt), 4), "val_f1": round(f1, 4)})
        if cetak:
            print(f"[{cetak}/{tag}] epoch {e:2d} loss {total / len(Xt):.4f} val_f1 {f1:.4f}", flush=True)
        if f1 > terbaik[0]:
            terbaik, bobot = (f1, e), {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bobot)
    return {"epoch_terbaik": terbaik[1], "riwayat": riwayat}


def jalankan(Xtr, ytr, Xva, yva, *, seed, cetak="", keluaran=None):
    """Latih TL dua tahap; kembalikan (model, info)."""
    Xtr = np.asarray(Xtr, np.float32)
    Xva = np.asarray(Xva, np.float32)
    n_kelas = keluaran or (int(np.max(ytr)) + 1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = buat(n_kelas, Xtr.shape[1])
    _statistik_norm(model, Xtr)

    # Tahap 1: bekukan badan, latih kepala saja
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    info_beku = _latih_tahap(model, Xtr, ytr, Xva, yva, seed=seed, epoch=HIPER["epoch_beku"],
                             lr=HIPER["lr"], wd=HIPER["wd"], batch=HIPER["batch"], cetak=cetak, tag="beku")

    # Tahap 2: buka layer3/layer4 + kepala, fine-tune
    for nama in HIPER["lapis_buka"]:
        for p in getattr(model, nama).parameters():
            p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    info_ft = _latih_tahap(model, Xtr, ytr, Xva, yva, seed=seed, epoch=HIPER["epoch"],
                           lr=HIPER["lr_ft"], wd=HIPER["wd"], batch=HIPER["batch"], cetak=cetak, tag="ft")

    info = {
        "status_pretrained": model.pretrained_status,
        "tahap1_epoch_terbaik": info_beku["epoch_terbaik"],
        "tahap2_epoch_terbaik": info_ft["epoch_terbaik"],
        "riwayat_beku": info_beku["riwayat"],
        "riwayat_finetune": info_ft["riwayat"],
    }
    return model, info
