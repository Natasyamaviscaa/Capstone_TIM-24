# Eksperimen RNN - NIM 123450011

## Model
- Arsitektur: Gated Recurrent Unit (GRU)
- Hidden size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Maksimum epoch: 50
- Early stopping: patience 8
- Seed eksperimen: 1
- Metrik utama: Macro F1

## Dataset
Target penelitian adalah klasifikasi tiga kondisi beban pompa:
1. Selang lurus
2. Selang tertekuk 90 derajat
3. Ujung selang dinaikkan 30 cm

## Status eksperimen
Implementasi model telah disiapkan. Pelatihan dan evaluasi belum dinyatakan selesai karena pemuat data pada `src/protokol/data.py` masih berupa template dan sumber pengukuran arus pompa belum terverifikasi.

Data `daya.csv` yang diperiksa memiliki kanal `kipas`, sehingga belum dapat dinyatakan sebagai sinyal arus pompa.

## Catatan
Metrik validasi hanya boleh dilaporkan setelah model dilatih menggunakan data yang sesuai dengan protokol tim dan hasil evaluasinya benar-benar diperoleh.
