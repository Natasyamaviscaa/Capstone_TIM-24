# Tim 24 — Diagnosis beban pompa dari sinyal listrik saja (kunci Sprint 1)

Repo tim yang sudah dikerjakan sampai akhir Sprint 1, dalam bentuk template `capstone-dl-2026-starter`
(@e8fdb7f). Angka di sini adalah satu hasil yang dapat dipertahankan pada data sesi timmu, bukan target.

**Ringkasnya:** sensor arus pompa tidak terpasang, jadi satu-satunya sinyal listrik yang tersisa adalah tegangan rel
5 V (`bus_mV`) yang ikut turun saat pompa menarik arus. Masalahnya, tegangan rel bergeser antarblok sebesar efek
beban itu sendiri: efek selang bertanda sama di kedua ulangan blok pada 0/8 pasangan daya, dan
dua blok berlabel sama dapat dibedakan dengan akurasi 0,81–0,99. Validasi (di dalam blok latih) memberi F1-makro
CNN 0,75 dan Transformer 0,74, tetapi angka itu sebagian besar "mengenali blok". Dugaan untuk blok uji:
mendekati tebakan (0,33). Pertanyaan kasus belum bisa dijawab dengan data ini.

## Menjalankan

1. Unduh arsip sesi `sesi-20261117-113416` dari kartu *Data meja*, ekstrak ke `data/sesi-20261117-113416/` (audio tidak dipakai).
2. Pasang dan jalankan:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.protokol.mutu          # reports/mutu-data.md (kanal, tegangan rel per blok, sidik blok)
python -m pytest -q
python run_all.py --seeds 1
python -m src.protokol.analisis      # reports/kesalahan.md (kebingungan, F1 per daya pompa)
```

## Hasil Sprint 1 (validasi di dalam blok latih, seed 1)

Validasi = ulangan 4 di blok latih 1-2, 1-3, 2-3 (287 jendela). Uji = blok 1-1, 2-1, 2-2
(850 jendela), belum disentuh.

| Jalur | F1-makro | F1 · pompa 120 / 160 / 200 / 255 | Parameter | Durasi (s) |
|---|---|---|---|---|
| Baseline starter (regresi logistik) | 0,56 | 0,56 / 0,64 / 0,66 / 0,25 | — | 0 |
| CNN 1-D | 0,75 | 0,66 / 0,84 / 0,53 / 0,86 | 43.747 | 18 |
| Transformer | 0,74 | 0,77 / 0,81 / 0,65 / 0,74 | 85.123 | 3 |
| Non-DL (statistik + FFT → SVM) | 0,68 | 0,64 / 0,81 / 0,46 / 0,75 | — | 0 |
| AE (encoder + kepala) | 0,63 | 0,50 / 0,69 / 0,57 / 0,53 | 31.523 | 16 |
| RNN (GRU) | 0,55 | 0,50 / 0,69 / 0,52 / 0,17 | 38.019 | 3 |

Cara membaca:
- **Validasi ini tidak mengukur beban.** Tiap keadaan selang punya satu blok latih, validasi dari blok yang sama.
  Kelas "ujung dinaikkan" hampir sempurna di CNN, Transformer, AE, dan non-DL (F1 0,93–0,97; `reports/kesalahan.md`),
  tetapi blok latihnya satu-satunya dari ulangan blok 2, yang tegangan relnya hingga ±60 mV lebih tinggi daripada dua
  blok latih lain pada daya 120–200. Model cukup belajar
  "tegangan tinggi → dinaikkan".
- **Lurus vs tertekuk** (dua blok latih dari ulangan blok yang sama) hanya F1 0,35–0,68: perbedaan beban yang
  sebenarnya sulit dilihat di tegangan rel.
- Per daya pompa tidak ada pola yang konsisten antarjalur (kartu kasus mengharapkan beda lebih jelas pada daya
  200 dan 255).
- RNN di bawah baseline (lihat catatannya).

## Keputusan utama (dan alternatif yang ditolak)

| Keputusan | Pilihan | Alternatif yang ditolak, dan alasannya |
|---|---|---|
| Sinyal | tegangan rel `bus_mV` (keputusan pembimbing: tetap listrik saja) | audio sebagai pengganti: mengubah pertanyaan kasus; menunggu sensor: Sprint 1 tanpa hasil |
| Satuan pemisahan | blok; blok dengan data lebih utuh untuk latih | per jendela/kondisi: blok yang sama di latih dan uji, angka tinggi palsu |
| Validasi | satu ulangan di blok latih, dengan uji sidik blok sebagai peringatan | validasi lintas blok: tidak mungkin tanpa memakai blok uji (2 blok per kelas) |
| Metrik | F1-makro + per daya pompa | gabungan saja (kartu: *Hindari*) |

## Temuan pada data tim ini

1. **INA219 pompa tidak terpasang** (kanal `daya.csv`: hanya kipas; arus kipas -0,1 mA
   karena kipas mati). Kartu kasus menulisnya sebagai prasyarat sebelum slot.
2. **Tegangan rel bergeser antarblok**: selang lurus bergeser +65, +53, +62, +4 mV antar-ulangan blok (per daya),
   sebesar atau lebih besar dari beda antar-keadaan selang. Efek selang konsisten pada 0/8
   pasangan.
3. **Sidik blok kuat** (0,81–0,99): validasi di dalam blok tidak bermakna untuk pertanyaan kasus.
4. **Wi-Fi ESP32 putus 12 kali**: 313 dari 2592 jendela
   ditolak, 12 kondisi dikeluarkan.

## Apa artinya untuk Sprint 2

Keputusan untuk pembimbing, sama seperti tim 8: desain ulang atau laporkan hasil negatif apa adanya.
- Desain ulang: pasang INA219 pompa (0x41), ulangi slot, dan acak keadaan selang di dalam blok (bukan satu keadaan
  per blok) supaya pergeseran antarblok tidak bertumpuk dengan label.
- Tanpa slot ulang: evaluasi blok uji sekali di sprint terakhir dan laporkan F1 per daya; tulis bahwa validasi
  Sprint 1 terkontaminasi sidik blok. Ablation yang berguna: hapus rata-rata tiap jendela (hanya riak) dan lihat
  apakah F1 validasi runtuh.

## Peta backlog Sprint 1 → bukti

| Tugas backlog | Bukti di repo ini |
|---|---|
| Rencana final disetujui dan diserahkan | `rencana.yaml` |
| Peran slot meja | `catatan/peran-slot.md` |
| Sprint Goal, Log Sprint, Demo | `sprint-goal.md`, `log-sprint-contoh.md`, `DEMO.md` |
| Pemotong jendela dan PROTOKOL.md dari data simulasi | `src/protokol/data.py`, `src/protokol/deret.py`, `src/protokol/PROTOKOL.md`, `tests/test_t24.py` |
| Terima data, MANIFEST.json, pemeriksaan mutu | `data/MANIFEST.json`, `src/protokol/mutu.py` → `reports/mutu-data.md` |
| Repo tim siap, run_all pada simulasi | `models/contoh-*/`, `models/non-dl/`, `run_all.py`, `catatan/gladi.md` |
| Merge komponen, run_all pada data primer | `reports/hasil_komparasi.md` |
| Pembanding non-DL | `models/non-dl/` (fitur statistik + FFT → Random Forest / SVM) |
| Gladi rencana | `catatan/gladi.md` |
| Rekaman Demo Sprint 1 | `DEMO.md` |
| [Jalur] Kerangka run.py dan latih seed 1 | `models/contoh-{cnn,rnn,trf,ae}/`, `results/*/metrics.json` |

## Yang berubah dari starter

- `run_all.py`: starter memanggil `src.common.baseline_mlp` yang tidak ada; diganti `src.common.baseline`.
- `src/protokol/deret.py`, `sidik.py` (baru). Folder model `contoh-<jalur>` pengganti NIM. `src/common/` tidak diubah.
