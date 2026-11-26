# Catatan model contoh-cnn (CNN 1-D)

**Hasil Sprint 1 (seed 1, validasi di dalam blok):** F1-makro 0,75 (per daya 120/160/200/255:
0,66 / 0,84 / 0,53 / 0,86); 43.747 parameter, 18 s.

## Bacaan
Tertinggi bersama Transformer, tetapi hampir seluruh keunggulannya dari kelas "ujung dinaikkan", yang bloknya
dikenali dari tingkat tegangan rel (README). Lurus vs tertekuk tetap lemah.

## Sprint 2
Ablation: kurangi rata-rata tiap jendela (hanya riak). Bila F1 runtuh, model memang hanya membaca tingkat tegangan.
