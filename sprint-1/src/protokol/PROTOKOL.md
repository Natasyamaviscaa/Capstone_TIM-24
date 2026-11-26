# Protokol evaluasi tim — Tim 24: Diagnosis beban pompa dari sinyal listrik saja

| Butir | Isi |
|---|---|
| Kasus | Mengenali keadaan selang pompa (lurus, tertekuk 90°, ujung dinaikkan 30 cm) hanya dari sinyal listrik |
| Sumber dataset | Sesi meja `sesi-20261117-113416` (17 Nov 2026, 2026-11-17 11:36–2026-11-17 12:10), arsip dari kartu *Data meja*; berkas + sha256 di `data/MANIFEST.json` |
| Jenis tugas (`TUGAS`) | klasifikasi 3 kelas |
| Label | keadaan selang dari faktor manual per blok; daya pompa (120/160/200/255) diacak di dalam blok, bukan label |
| Masukan | **tegangan rel 5 V** (`bus_mV`, INA219 kipas), jendela 1 s bergeser 0,5 s, detik pertama dibuang, 200 Hz menurut jam ESP32 → `[1, 200]` |
| Penyimpangan dari rencana | Rencana meminta arus pompa (INA219 0x41) yang tidak terpasang; kipas mati, jadi arus kipas ±0. Pompa terlihat tidak langsung lewat turunnya tegangan rel bersama (pola yang sama dengan kunci tim 23). Tetap "sinyal listrik saja" |
| Jendela ditolak / kondisi dikeluarkan | 313 jendela bercelah (Wi-Fi ESP32 putus 12 kali); 12 kondisi yang daya pompanya tidak sesuai label |
| Satuan pemisahan | **Blok**: tiap keadaan selang punya 2 blok; blok dengan jendela sah lebih banyak untuk latih + validasi (blok 1-2, 1-3, 2-3), yang lain uji (blok 1-1, 2-1, 2-2). Validasi = ulangan 4 di blok latih |
| Ukuran latih / validasi / uji | 818 / 287 / 850 jendela |
| Pembanding | baseline starter (regresi logistik); non-DL fitur statistik + FFT + Random Forest/SVM (rencana tim) |
| Metrik utama | F1-makro, juga per daya pompa (kartu: jangan hanya gabungan) |
| Praproses | normalisasi per kanal dari data latih saja |
| Anggaran komputasi | ≤ 20 menit CPU laptop per jalur per seed |
| Evaluasi data uji | Sekali, sprint terakhir |
| `data_hash` | `9c283bca03ef20a1840af2bae31ecf48` |

## Batas validasi ini: sidik blok

Tiap keadaan selang hanya punya satu blok latih, dan validasi diambil dari blok yang sama. Dua blok berlabel sama
dapat dibedakan dari datanya dengan akurasi 0,81–0,99 (`reports/mutu-data.md`), dan tegangan rel bergeser antarblok
sebesar efek beban. Jadi F1 validasi mengukur "mengenali blok", bukan "mengenali beban". Ukuran yang jujur baru
datang dari blok uji (sekali, sprint terakhir), dan dugaan dari data adalah: mendekati tebakan.
