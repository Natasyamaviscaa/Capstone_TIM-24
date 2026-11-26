"""Pemuat data tim 24: diagnosis beban pompa dari sinyal listrik saja.

Kontrak starter: ``TUGAS`` dan ``muat() -> X_train, y_train, X_val, y_val, X_test, y_test``.

Keputusan protokol (rinciannya di PROTOKOL.md):
- **Penyimpangan dari rencana:** rencana meminta arus pompa (INA219 0x41), tetapi modul itu tidak terpasang;
  ``daya.csv`` hanya berisi kanal ``kipas`` (INA219 0x40), dan kipas mati sepanjang sesi (arus ±0). Pompa berbagi
  catu 5 V dengan INA219 kipas, jadi beban pompa hanya terlihat TIDAK LANGSUNG lewat tegangan rel (``bus_mV``),
  seperti jejak pompa di kunci tim 23. Masukan = ``bus_mV`` saja, tetap "sinyal listrik saja".
- Label = keadaan selang (lurus, tertekuk 90°, ujung dinaikkan 30 cm) dari faktor manual per blok; daya pompa
  (120/160/200/255) diacak di dalam blok dan BUKAN label.
- Satu contoh = jendela 1 s bergeser 0,5 s (detik pertama kondisi dibuang), dicuplik ulang ke 200 Hz memakai jam
  ESP32 → X [N, 1, 200]. Jendela dengan celah sampel > 50 ms (Wi-Fi putus) ditolak; kondisi yang daya pompanya
  menurut ``aktuator.csv`` berbeda dari label dikeluarkan.
- Satuan pemisahan = blok (ulangan_blok, blok). Tiap keadaan selang punya 2 blok: blok dengan jendela sah LEBIH
  BANYAK untuk latih + validasi, yang lebih sedikit untuk uji (ditetapkan dari jumlah data, sebelum melatih).
  Validasi = satu ulangan kondisi (dari 4, ``split.indeks`` seed 2026) di blok latih.
- Normalisasi per kanal dari data latih saja.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict

import numpy as np

from src.common import manifest, split
from src.protokol import deret, pisah, sesi

TUGAS = "klasifikasi"
KELAS = ["selang lurus", "selang tertekuk 90 derajat", "ujung selang dinaikkan 30 cm"]
METRIK_TIM = ("val_f1_makro", "F1-makro")

SESI = "sesi-20261117-113416"
LAJU = 200.0
JENDELA_S = 1.0
LANGKAH_S = 0.5
BUANG_AWAL_S = 1.0
KANAL = ["bus_mV"]
CEK_AKTUATOR_S = (1.0, 7.0, 14.0)
VERSI_FITUR = "t24-busmV-200hz-v1"

FOLDER = manifest.AKAR / "data" / SESI
CACHE = manifest.AKAR / "data" / "cache"


def metrik_tim(y: np.ndarray, p: np.ndarray, logit: np.ndarray | None = None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    per = {}
    for i, nama in enumerate(KELAS):
        tp = int(((p == i) & (y == i)).sum()); fp = int(((p == i) & (y != i)).sum()); fn = int(((p != i) & (y == i)).sum())
        per[nama] = round(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn), 4)
    return {"val_f1_makro": round(float(np.mean(list(per.values()))), 4), "val_f1_rinci": per}


def satuan_blok(b: dict) -> int:
    return int(b["ulangan_blok"]) * 10 + int(b["blok"])


def kondisi_dibuang(label: list[dict]) -> dict[int, str]:
    per = defaultdict(list)
    for r in sesi.baca_csv(FOLDER, "aktuator.csv"):
        per[r["nama"]].append((float(r["t_host"]), float(r["nilai"])))
    waktu = [t for t, _ in per["pompa"]]
    buang = {}
    for b in label:
        for off in CEK_AKTUATOR_S:
            i = bisect.bisect_right(waktu, b["t_mulai_host"] + off) - 1
            if i < 0 or per["pompa"][i][1] != float(b["pompa"]):
                buang[int(b["no"])] = "daya pompa tidak sesuai label menurut aktuator.csv"
                break
    return buang


def baca_listrik() -> deret.Deret:
    return deret.baca(FOLDER, "daya.csv", KANAL, saring=("kanal", "kipas"))


def _sidik() -> str:
    h = hashlib.sha256(VERSI_FITUR.encode())
    for nama in ("label.csv", "aktuator.csv"):
        h.update((FOLDER / nama).read_bytes())
    h.update(str((FOLDER / "daya.csv").stat().st_size).encode())
    h.update(json.dumps([LAJU, JENDELA_S, LANGKAH_S, BUANG_AWAL_S, CEK_AKTUATOR_S, KANAL]).encode())
    return h.hexdigest()[:16]


def contoh() -> dict:
    """Semua jendela sah (belum dinormalisasi), di-cache. Kunci: X [N, 1, 200], y, satuan, ulangan, kondisi, pompa,
    dibuang (json), jendela_ditolak (json {no: jumlah})."""
    if not FOLDER.exists():
        raise FileNotFoundError(f"data/{SESI}/ belum ada: unduh arsip sesi tim dari kartu Data meja lalu ekstrak "
                                "ke data/ (lihat data/MANIFEST.json).")
    berkas = CACHE / f"{_sidik()}.npz"
    if berkas.exists():
        with np.load(berkas) as z:
            return {k: z[k] for k in z.files}
    label = sesi.baca_label(FOLDER)
    buang = kondisi_dibuang(label)
    listrik = baca_listrik()
    kolom = {k: [] for k in ("X", "y", "satuan", "ulangan", "kondisi", "pompa")}
    ditolak: dict[int, int] = {}
    for b in label:
        no = int(b["no"])
        if no in buang:
            continue
        t = b["t_mulai_host"] + BUANG_AWAL_S
        while t + JENDELA_S <= b["t_selesai_host"] + 1e-6:
            g = deret.potong(listrik, t, t + JENDELA_S, LAJU)
            if g is None:
                ditolak[no] = ditolak.get(no, 0) + 1
            else:
                kolom["X"].append(g)
                kolom["y"].append(KELAS.index(b["beban_pompa"]))
                kolom["satuan"].append(satuan_blok(b))
                kolom["ulangan"].append(int(b["ulangan"]))
                kolom["kondisi"].append(no)
                kolom["pompa"].append(int(float(b["pompa"])))
            t += LANGKAH_S
    hasil = {k: np.asarray(v, np.float32 if k == "X" else None) for k, v in kolom.items()}
    hasil["dibuang"] = np.array(json.dumps(buang, ensure_ascii=False))
    hasil["jendela_ditolak"] = np.array(json.dumps(ditolak))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(berkas, **hasil)
    return hasil


def bagian() -> dict:
    c = contoh()
    jumlah = defaultdict(lambda: defaultdict(int))
    for s, y in zip(c["satuan"].tolist(), c["y"].tolist()):
        jumlah[y][s] += 1
    uji = []
    for y in sorted(jumlah):
        dua = sorted(jumlah[y], key=lambda s: (jumlah[y][s], s))
        uji += dua[:-1] if len(dua) > 1 else []
    latih_blok = sorted(set(c["satuan"].tolist()) - set(uji))
    _, val_idx = split.indeks(4, porsi_val=0.25)
    ulangan_val = sorted(int(i) + 1 for i in val_idx)
    m_uji = pisah.topeng(c["satuan"], uji)
    m_val = ~m_uji & np.isin(c["ulangan"], ulangan_val)
    return {"c": c, "blok": {"latih": latih_blok, "uji": sorted(uji)}, "ulangan_val": ulangan_val,
            "latih": ~m_uji & ~m_val, "val": m_val, "uji": m_uji}


def muat(standar: bool = True):
    """X [N, 1, 200] float32 (dinormalisasi dari data latih bila ``standar``), y int64 0–2."""
    b = bagian(); c = b["c"]
    pisah.periksa_tidak_bocor(c["satuan"][b["latih"] | b["val"]], c["satuan"][b["uji"]])
    X, y = c["X"], c["y"].astype(np.int64)
    if standar:
        X = deret.StandarKanal(X[b["latih"]])(X)
    return (X[b["latih"]], y[b["latih"]], X[b["val"]], y[b["val"]], X[b["uji"]], y[b["uji"]])
