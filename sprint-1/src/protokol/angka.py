"""Kumpulkan angka laporan → reports/angka.json.

    python -m src.protokol.angka
"""
from __future__ import annotations

import json

from src.common import manifest, split
from src.protokol import data

KODE = {"baseline": "BASE", "contoh-cnn": "CNN", "contoh-rnn": "RNN", "contoh-trf": "TRF", "contoh-ae": "AE", "non-dl": "NONDL"}


def _f(x, d: int = 2) -> str:
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def main() -> None:
    akar = manifest.AKAR
    a: dict = {}
    mutu = json.loads((akar / "reports" / "mutu.json").read_text("utf-8"))
    for k, v in mutu.items():
        if not isinstance(v, (dict, list)):
            a[f"MUTU_{k}"] = str(v).replace(".", ",") if isinstance(v, float) else v
    a["GESER_LURUS"] = ", ".join(f"{v:+d}" for v in mutu["geser_antar_ub_lurus"])
    s = [v for v in mutu["sidik_blok"].values() if v is not None]
    a["SIDIK"] = f"{_f(min(s))}–{_f(max(s))}" if s else "—"
    b = data.bagian()
    a["BLOK_UJI"] = ", ".join(f"{x // 10}-{x % 10}" for x in b["blok"]["uji"])
    a["BLOK_LATIH"] = ", ".join(f"{x // 10}-{x % 10}" for x in b["blok"]["latih"])
    a["ULANGAN_VAL"] = ", ".join(map(str, b["ulangan_val"]))
    for n in ("latih", "val", "uji"):
        a[f"N_{n.upper()}"] = int(b[n].sum())
    a["SEED_SPLIT"] = split.SEED_SPLIT
    kes = json.loads((akar / "reports" / "kesalahan.json").read_text("utf-8")) if (akar / "reports" / "kesalahan.json").exists() else {}
    for nama, r in kes.items():
        k = KODE.get(nama, nama.upper())
        for pw, v in r["f1_per_daya"].items():
            a[f"{k}_F1_P{pw}"] = _f(v)
    for d in sorted((akar / "results").iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text("utf-8"))
        t = json.loads((d / "tambahan.json").read_text("utf-8")) if (d / "tambahan.json").exists() else {}
        k = KODE.get(d.name, d.name.upper())
        a[f"{k}_ACC"] = _f(m["metrik"]["val_acc"]); a[f"{k}_F1"] = _f(m["metrik"]["val_macro_f1"])
        a[f"{k}_DURASI"] = f"{m.get('durasi_detik', 0):.0f}"
        a["DATA_HASH"] = m["data_hash"]
        if "parameter" in t:
            a[f"{k}_PARAM"] = f"{t['parameter']:,}".replace(",", ".")
        if k == "NONDL" and "dipilih" in t:
            a["NONDL_PILIH"] = {"rf": "Random Forest", "svm": "SVM"}.get(t["dipilih"], t["dipilih"])
    (akar / "reports" / "angka.json").write_text(json.dumps(a, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"{len(a)} angka → reports/angka.json")


if __name__ == "__main__":
    main()
