# Peran slot meja (contoh)

Rencana tim 24: 3 keadaan selang (manual per blok, 2 ulangan blok) × daya pompa 120/160/200/255 × 4 ulangan,
15 s per kondisi (96 kondisi, ±39 menit). Nama anggota tidak ditulis di repo.

| Peran | Tugas pada slot | Selesai bila |
|---|---|---|
| Operator perekam | **Sebelum slot**: kartu Arus pompa ada di halaman meja; bila tidak, slot ditunda | Kanal `pompa` tercatat di `daya.csv` menit pertama |
| Pengatur selang | Tekukan di titik bertanda dengan penjepit; penyangga 30 cm; air dikembalikan ke garis acuan setiap blok | Tinggi air dicatat per blok |
| Pengawas sambungan | Memantau "Wi-Fi ke ESP32 putus" | Kondisi terdampak dicatat |
| Penjaga data | Menjalankan `python -m src.protokol.mutu` pada hari yang sama | `reports/mutu-data.md` di-commit |

## Yang terjadi pada sesi tim 24 (pelajaran)

Kartu kasus menulis prasyarat INA219 pompa; sesi tetap berjalan tanpanya. Data listrik yang tersisa (tegangan rel)
bergeser antarblok sebesar efek beban, sehingga hasil Sprint 1 tidak bisa menjawab pertanyaan kasus. Pemeriksaan
prasyarat sebelum slot lebih murah daripada slot ulang.
