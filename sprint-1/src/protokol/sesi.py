"""Membaca satu folder sesi Meja 5 Volt (keluaran ``meja.py jalankan``).

Isi folder sesi yang dipakai di sini::

    label.csv          satu baris per kondisi: faktor rencana + t_mulai_host, t_selesai_host (detik epoch)
    audio_index.csv    berkas, t_mulai_host, sample_rate, jumlah_sampel (satu baris per potongan WAV)
    audio/*.wav        rekaman bersambung, dipotong per 60 detik
    daya.csv, imu.csv, aktuator.csv, lingkungan.csv   sensor ESP32 (t_host = jam komputer meja)
    manifest.json      sha256 tiap berkas (ditulis saat sesi ditutup)

Semua waktu memakai jam komputer meja (``t_host``), sehingga label dan audio dapat dipotong
tanpa menebak. Sejak perekam 1.7 audio disimpan pada laju asli mikrofon (48 kHz); ``Audio.potong``
menurunkan ke laju yang diminta.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path


def baca_label(folder: Path) -> list[dict]:
    """Baris label.csv; t_mulai_host/t_selesai_host menjadi float."""
    with (Path(folder) / "label.csv").open(encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    for b in baris:
        b["t_mulai_host"] = float(b["t_mulai_host"])
        b["t_selesai_host"] = float(b["t_selesai_host"])
    return baris


def baca_csv(folder: Path, nama: str) -> list[dict]:
    p = Path(folder) / nama
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def periksa_manifest(folder: Path, hanya: tuple[str, ...] = ()) -> list[str]:
    """Bandingkan sha256 berkas dengan manifest.json sesi. Kembalikan daftar berkas yang beda/hilang."""
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["berkas"]
    masalah = []
    for nama, info in m.items():
        if nama == "log.txt" or (hanya and not nama.startswith(hanya)):
            continue
        p = folder / nama
        if not p.exists():
            masalah.append(f"hilang: {nama}")
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for blok in iter(lambda: f.read(1 << 20), b""):
                h.update(blok)
        if h.hexdigest() != info["sha256"]:
            masalah.append(f"beda: {nama}")
    return masalah


@dataclass
class Potongan:
    berkas: Path
    t0: float
    sr: int
    n: int

    @property
    def t1(self) -> float:
        return self.t0 + self.n / self.sr


class Audio:
    """Rekaman bersambung satu sesi; potong berdasarkan jam komputer meja."""

    def __init__(self, folder: Path):
        folder = Path(folder)
        self.potongan = [Potongan(folder / "audio" / b["berkas"], float(b["t_mulai_host"]), int(b["sample_rate"]),
                                  int(b["jumlah_sampel"]))
                         for b in baca_csv(folder, "audio_index.csv")]
        self.potongan.sort(key=lambda p: p.t0)

    def celah(self) -> list[float]:
        """Selisih (detik) antara akhir satu potongan dan awal potongan berikutnya (positif = celah)."""
        return [b.t0 - a.t1 for a, b in zip(self.potongan, self.potongan[1:])]

    def potong(self, t_awal: float, t_akhir: float, sr: int = 16000) -> np.ndarray | None:
        """Sinyal float32 [-1, 1] antara dua waktu host, diturunkan ke ``sr``. None bila tidak tertutup audio."""
        import numpy as np
        import soundfile as sf

        bagian, t = [], t_awal
        for p in self.potongan:
            if p.t1 <= t or p.t0 >= t_akhir:
                continue
            if p.t0 - t > 0.05:            # celah > 50 ms di tengah jendela: jendela ditolak
                return None
            a = max(0, int(round((t - p.t0) * p.sr)))
            b = min(p.n, int(round((t_akhir - p.t0) * p.sr)))
            x, sr_asli = sf.read(p.berkas, start=a, stop=b, dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr_asli != sr:
                from scipy.signal import resample_poly
                g = gcd(sr, sr_asli)
                x = resample_poly(x, sr // g, sr_asli // g).astype(np.float32)
            bagian.append(x)
            t = p.t0 + b / p.sr
            if t >= t_akhir - 1e-3:
                break
        if not bagian or t < t_akhir - 0.05:
            return None
        x = np.concatenate(bagian)
        n = int(round((t_akhir - t_awal) * sr))
        return np.pad(x, (0, max(0, n - len(x))))[:n]
