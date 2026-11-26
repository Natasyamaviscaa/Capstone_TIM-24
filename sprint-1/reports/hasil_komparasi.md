# Hasil komparasi (dibangkitkan run_all.py)

| Folder | Model | seed | val_acc | val_macro_f1 | F1-makro | parameter | latensi (ms) | durasi (s) |
|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 1 | 0.5645 | 0.5577 | — | — | — | 0 |
| contoh-ae | ae | 1 | 0.6376 | 0.6314 | 0.6313 | 31523 | 0.17 | 16 |
| contoh-cnn | cnn | 1 | 0.7422 | 0.7474 | 0.7473 | 43747 | 0.26 | 18 |
| contoh-rnn | rnn | 1 | 0.5436 | 0.5518 | 0.5518 | 38019 | 0.4 | 3 |
| contoh-trf | trf | 1 | 0.7387 | 0.7430 | 0.7430 | 85123 | 0.28 | 3 |
| non-dl | non-dl-svm | 1 | 0.6725 | 0.6757 | 0.6757 | — | — | 0 |
