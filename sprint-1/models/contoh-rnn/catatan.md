# Catatan model contoh-rnn (GRU)

**Hasil Sprint 1 (seed 1, validasi di dalam blok):** F1-makro 0,55 (per daya 120/160/200/255:
0,50 / 0,69 / 0,52 / 0,17); 38.019 parameter, 3 s.

## Di bawah baseline starter
Sedikit di bawah baseline starter (F1-makro 0,56), dan bahkan gagal memakai sidik tingkat tegangan
sebaik jalur lain. GRU atas 200 langkah satu kanal yang hampir datar (riak kecil di atas tingkat DC) tidak punya
pola waktu yang berguna. Dengan data yang tidak membawa beban, peringkat jalur
di Sprint 1 tidak bermakna.

## Sprint 2
Ringkas jendela (mis. statistik per 50 ms) sebelum GRU, atau lewati jalur ini bila keputusan desain ulang diambil.
