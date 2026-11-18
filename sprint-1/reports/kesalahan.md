# Analisis kesalahan (validasi, seed 1)

Validasi = ulangan [4] di blok latih (287 jendela). Validasi berada DI DALAM blok latih: lihat sidik blok di mutu-data.

| | baseline starter | AE | CNN | RNN | Transformer | non-DL |
|---|---|---|---|---|---|---|
| **F1-makro** | **0.56** | **0.63** | **0.75** | **0.55** | **0.74** | **0.68** |
| F1 selang lurus | 0.39 | 0.35 | 0.65 | 0.35 | 0.62 | 0.46 |
| F1 selang tertekuk 90 derajat | 0.56 | 0.57 | 0.63 | 0.52 | 0.68 | 0.61 |
| F1 ujung selang dinaikkan 30 cm | 0.73 | 0.97 | 0.96 | 0.78 | 0.93 | 0.96 |
| F1-makro · pompa 120 | 0.56 | 0.50 | 0.66 | 0.50 | 0.77 | 0.64 |
| F1-makro · pompa 160 | 0.64 | 0.69 | 0.84 | 0.69 | 0.81 | 0.81 |
| F1-makro · pompa 200 | 0.66 | 0.57 | 0.53 | 0.52 | 0.65 | 0.46 |
| F1-makro · pompa 255 | 0.25 | 0.53 | 0.86 | 0.17 | 0.74 | 0.75 |

### baseline starter

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 31 | 48 | 10 |
| tertekuk | 27 | 65 | 13 |
| dinaikkan | 12 | 15 | 66 |

### AE

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 28 | 60 | 1 |
| tertekuk | 38 | 67 | 0 |
| dinaikkan | 3 | 2 | 88 |

### CNN

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 69 | 19 | 1 |
| tertekuk | 48 | 57 | 0 |
| dinaikkan | 6 | 0 | 87 |

### RNN

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 29 | 58 | 2 |
| tertekuk | 39 | 66 | 0 |
| dinaikkan | 8 | 24 | 61 |

### Transformer

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 57 | 30 | 2 |
| tertekuk | 33 | 71 | 1 |
| dinaikkan | 6 | 3 | 84 |

### non-DL

| benar \ tebakan | lurus | tertekuk | dinaikkan |
|---|---|---|---|
| lurus | 40 | 49 | 0 |
| tertekuk | 38 | 67 | 0 |
| dinaikkan | 7 | 0 | 86 |
