
# Catatan Model 123450020

## Jalur Model
CNN 1D untuk klasifikasi kondisi selang pompa berdasarkan sinyal listrik bus_mV.

## Alasan Optimizer
Menggunakan Adam dengan learning rate 0.002 untuk mengoptimalkan pembaruan bobot model CNN.

## Regularisasi
Menggunakan Batch Normalization dan Dropout 0.2 untuk membantu mengendalikan overfitting.

## Pencarian Hyperparameter
Arsitektur CNN 1D:
- Convolutional channels: 32, 64, 64
- Kernel size: 7
- Optimizer: Adam
- Learning rate: 0.002
- Batch size: 64
- Epoch: 25
- Weight decay: 0.0001

## Ablation
1. Baseline CNN 1D.
2. CNN 1D dengan penghilangan mean per jendela (mean removal).

## Hasil 3 Seed Konfigurasi Akhir
Eksperimen lanjutan lima seed sudah dilakukan pada Sprint 1 sebagai analisis tambahan. Ringkasan hasil lima seed tercatat dalam laporan individu.

Hasil eksekusi khusus folder anggota:
- Seed: 1
- Validation Macro F1: 0.747608
- Validation Accuracy: 0.742160
- Data hash: 9c283bca03ef20a1840af2bae31ecf48
