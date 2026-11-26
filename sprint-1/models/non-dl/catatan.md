# Catatan pembanding non-deep learning (fitur statistik + FFT)

**Hasil Sprint 1 (seed 1, validasi di dalam blok):** F1-makro 0,68 (SVM; per daya
120/160/200/255: 0,64 / 0,81 / 0,46 / 0,75).

## Fitur dan pelatihan
Rata-rata, simpangan, RMS, puncak, kemencengan, keruncingan, dan energi 8 pita FFT dari `bus_mV` (rencana tim);
Random Forest dan SVM dibandingkan dengan F1-makro validasi.

## Bacaan
Fitur "rata-rata" adalah tingkat tegangan rel yang membawa sidik blok. Spektrum riak memuat puncak ±74,5 Hz pada
daya < 255 (lipatan frekuensi PWM, bergantung daya, bukan beban).

## Sprint 2
Kepentingan fitur: bila "rata-rata" mendominasi, itu bukti tambahan bahwa model membaca blok.
