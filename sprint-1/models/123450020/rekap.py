
import json
import statistics
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]

EKSPERIMEN = {
    "CNN Baseline": "contoh-cnn",
    "CNN Ablation Riak": "contoh-cnn-ablation-riak",
}

METRIK = ["val_acc", "val_macro_f1"]

print("=" * 65)
print("REKAP HASIL EKSPERIMEN CNN - 5 SEED")
print("=" * 65)

for nama, folder in EKSPERIMEN.items():
    print(f"\n{nama}")
    print("-" * 65)

    hasil = {m: [] for m in METRIK}

    for seed in range(1, 6):
        file = AKAR / "results" / folder / f"metrics-seed{seed}.json"

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrik = data["metrik"]

        print(f"Seed {seed}: ", end="")

        for m in METRIK:
            nilai = metrik[m]
            hasil[m].append(nilai)
            print(f"{m}={nilai:.6f}  ", end="")

        print()

    print("\nRINGKASAN:")
    for m, nilai in hasil.items():
        rata = statistics.mean(nilai)
        std = statistics.stdev(nilai)

        print(f"{m}:")
        print(f"  Rata-rata       = {rata:.6f}")
        print(f"  Standar deviasi = {std:.6f}")

print("\n" + "=" * 65)
