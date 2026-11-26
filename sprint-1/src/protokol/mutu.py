"""Pemeriksaan mutu data sesi tim 24 → reports/mutu-data.md + reports/mutu.json

    python -m src.protokol.mutu

Umum: kelengkapan (96 = 6 blok × 4 daya × 4 ulangan), keutuhan berkas, sambungan ESP32, jendela ditolak. Khusus
tim 24: kanal listrik yang tersedia, konsistensi efek beban selang pada tegangan rel antar-blok (per daya pompa),
dan uji sidik blok (dua blok berlabel sama dapat dibedakan?).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from src.common import manifest
from src.protokol import data, deret, sesi, sidik

SINGKAT = {"selang lurus": "lurus", "selang tertekuk 90 derajat": "tertekuk", "ujung selang dinaikkan 30 cm": "dinaikkan"}


def main() -> None:
    folder = data.FOLDER
    label = sesi.baca_label(folder)
    log = (folder / "log.txt").read_text(encoding="utf-8").strip().splitlines()
    h: dict = {}
    temuan: list[str] = []
    with (folder / "daya.csv").open(encoding="utf-8") as f:
        kanal = Counter(r["kanal"] for r in csv.DictReader(f))
    c = data.contoh()
    ditolak = json.loads(str(c["jendela_ditolak"])); dibuang = json.loads(str(c["dibuang"]))
    h.update(n_kondisi=len(label), n_blok=len({data.satuan_blok(b) for b in label}), kanal_daya=dict(kanal),
             log_selesai=log[-1].split("  ", 1)[-1].startswith("Selesai"),
             mulai=label[0]["mulai_iso"][:16].replace("T", " "), selesai=log[-1][:16].replace("T", " "),
             berkas_bermasalah=len(sesi.periksa_manifest(folder, ("label.csv", "daya.csv", "aktuator.csv", "rencana.yaml"))),
             wifi_putus=sum("Wi-Fi ke ESP32 putus" in r for r in log),
             n_dibuang=len(dibuang), jendela_sah=int(len(c["y"])), jendela_ditolak=int(sum(ditolak.values())),
             jendela_rencana=len(label) * int((15 - data.BUANG_AWAL_S - data.JENDELA_S) / data.LANGKAH_S + 1))
    ar = deret.baca(folder, "daya.csv", ["arus_mA"], saring=("kanal", "kipas")).X[:, 0]
    h["arus_kipas_median"] = round(float(np.median(ar)), 2)
    # tegangan rel per keadaan selang × daya × ulangan blok (median jendela)
    rel = c["X"][:, 0].mean(1)
    ub = c["satuan"] // 10
    tab = defaultdict(dict)
    for k, nama in enumerate(data.KELAS):
        for pw in sorted(set(c["pompa"].tolist())):
            for u in (1, 2):
                m = (c["y"] == k) & (c["pompa"] == pw) & (ub == u)
                tab[f"{SINGKAT[nama]} · pompa {pw}"][f"ub{u}"] = round(float(np.median(rel[m])), 0) if m.any() else None
    h["rel_mV"] = tab
    # efek beban relatif terhadap selang lurus, per daya dan ulangan blok: tandanya sama di kedua ulangan blok?
    konsisten = total = 0
    for nama in ("tertekuk", "dinaikkan"):
        for pw in sorted(set(c["pompa"].tolist())):
            e = [None if tab[f"{nama} · pompa {pw}"][f"ub{u}"] is None or tab[f"lurus · pompa {pw}"][f"ub{u}"] is None
                 else tab[f"{nama} · pompa {pw}"][f"ub{u}"] - tab[f"lurus · pompa {pw}"][f"ub{u}"] for u in (1, 2)]
            if None not in e:
                total += 1; konsisten += int(np.sign(e[0]) == np.sign(e[1]) and abs(e[0]) > 10 and abs(e[1]) > 10)
    h["efek_konsisten"] = f"{konsisten}/{total}"
    lurus = [tab[f"lurus · pompa {pw}"] for pw in sorted(set(c["pompa"].tolist()))]
    h["geser_antar_ub_lurus"] = [round(v["ub2"] - v["ub1"]) for v in lurus if None not in v.values()]
    # sidik blok
    F = deret.fitur_statistik(c["X"], data.LAJU)
    h["sidik_blok"] = sidik.uji_sidik(F, c["y"], c["satuan"], c["kondisi"], data.KELAS)
    (manifest.AKAR / "reports").mkdir(exist_ok=True)
    (manifest.AKAR / "reports" / "mutu.json").write_text(json.dumps(h, indent=2, ensure_ascii=False) + "\n", "utf-8")

    temuan.append(f"**Sensor arus pompa tidak terpasang.** `daya.csv` hanya berisi kanal {', '.join(kanal)}; arus kipas "
                  f"median {h['arus_kipas_median']} mA (kipas mati sepanjang sesi). Satu-satunya sinyal listrik yang "
                  "terkait pompa adalah tegangan rel 5 V (`bus_mV`) yang turun saat pompa menarik arus.")
    temuan.append(f"Efek keadaan selang pada tegangan rel (relatif terhadap selang lurus, per daya) bertanda sama dan "
                  f"> 10 mV di kedua ulangan blok hanya pada {h['efek_konsisten']} pasangan; tegangan rel selang lurus "
                  f"sendiri bergeser {', '.join(f'{v:+d}' for v in h['geser_antar_ub_lurus'])} mV antar-ulangan blok "
                  "(per daya). Pergeseran antarblok sebesar atau lebih besar dari efek beban.")
    nilai = [v for v in h["sidik_blok"].values() if v is not None]
    if nilai and min(nilai) > 0.8:
        temuan.append(f"Sidik blok kuat: dua blok berlabel sama dibedakan dengan akurasi {min(nilai):.2f}–{max(nilai):.2f}. "
                      "Validasi di dalam blok latih akan tampak bagus walau model hanya mengenali blok.")
    temuan.append(f"Wi-Fi ESP32 putus {h['wifi_putus']} kali: {h['jendela_ditolak']} jendela ditolak karena celah sampel, "
                  f"{h['n_dibuang']} kondisi dikeluarkan karena daya pompa tidak sesuai label.")
    baris = [
        f"# Mutu data sesi {data.SESI}", "", "Dibangkitkan `python -m src.protokol.mutu`; angka mentah di `reports/mutu.json`.", "",
        "## Kelengkapan", "", "| Pemeriksaan | Hasil | Harapan |", "|---|---|---|",
        f"| Jumlah kondisi / blok | {h['n_kondisi']} / {h['n_blok']} | 96 / 6 |",
        f"| Kanal `daya.csv` | {', '.join(f'{k} ({v} baris)' for k, v in kanal.items())} | pompa (dan kipas) |",
        f"| Jendela sah / rencana | {h['jendela_sah']} / {h['jendela_rencana']} | semua |",
        f"| Kondisi dikeluarkan (daya pompa ≠ label) | {h['n_dibuang']} | 0 |",
        f"| Wi-Fi ESP32 putus | {h['wifi_putus']} kali | 0 |",
        f"| Log berakhir 'Selesai' | {'ya' if h['log_selesai'] else 'TIDAK'} | ya |",
        f"| Berkas yang dipakai cocok dengan manifest.json | {'semua' if not h['berkas_bermasalah'] else h['berkas_bermasalah']} | semua |",
        "", "## Tegangan rel (mV, median jendela) per keadaan selang, daya pompa, dan ulangan blok", "",
        "| Keadaan · daya | ulangan blok 1 | ulangan blok 2 |", "|---|---|---|",
        *[f"| {k} | {v['ub1']:.0f} | {v['ub2']:.0f} |" if None not in v.values() else f"| {k} | {v['ub1']} | {v['ub2']} |"
          for k, v in sorted(tab.items())],
        "", "## Sidik blok (dua blok berlabel sama dibedakan dari fitur statistik, validasi silang per kondisi)", "",
        "Mendekati 1 = blok punya sidik kuat; 0,5 = tidak terbedakan. Pemeriksaan data, bukan evaluasi model.", "",
        "| Keadaan selang | akurasi |", "|---|---|",
        *[f"| {k} | {v if v is not None else 'data blok terlalu sedikit'} |" for k, v in h["sidik_blok"].items()],
        "", "## Temuan", "", *(f"- {t}" for t in temuan), "",
    ]
    (manifest.AKAR / "reports" / "mutu-data.md").write_text("\n".join(baris), encoding="utf-8")
    print(json.dumps(h, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
