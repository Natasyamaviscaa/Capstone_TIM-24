# Mutu data sesi sesi-20261117-113416

Dibangkitkan `python -m src.protokol.mutu`; angka mentah di `reports/mutu.json`.

## Kelengkapan

| Pemeriksaan | Hasil | Harapan |
|---|---|---|
| Jumlah kondisi / blok | 96 / 6 | 96 / 6 |
| Kanal `daya.csv` | kipas (342710 baris) | pompa (dan kipas) |
| Jendela sah / rencana | 1955 / 2592 | semua |
| Kondisi dikeluarkan (daya pompa ≠ label) | 12 | 0 |
| Wi-Fi ESP32 putus | 12 kali | 0 |
| Log berakhir 'Selesai' | ya | ya |
| Berkas yang dipakai cocok dengan manifest.json | semua | semua |

## Tegangan rel (mV, median jendela) per keadaan selang, daya pompa, dan ulangan blok

| Keadaan · daya | ulangan blok 1 | ulangan blok 2 |
|---|---|---|
| dinaikkan · pompa 120 | 4836 | 4824 |
| dinaikkan · pompa 160 | 4803 | 4789 |
| dinaikkan · pompa 200 | 4789 | 4781 |
| dinaikkan · pompa 255 | 4740 | 4730 |
| lurus · pompa 120 | 4762 | 4827 |
| lurus · pompa 160 | 4747 | 4800 |
| lurus · pompa 200 | 4717 | 4779 |
| lurus · pompa 255 | 4725 | 4729 |
| tertekuk · pompa 120 | 4761 | 4818 |
| tertekuk · pompa 160 | 4736 | 4793 |
| tertekuk · pompa 200 | 4717 | 4776 |
| tertekuk · pompa 255 | 4674 | 4727 |

## Sidik blok (dua blok berlabel sama dibedakan dari fitur statistik, validasi silang per kondisi)

Mendekati 1 = blok punya sidik kuat; 0,5 = tidak terbedakan. Pemeriksaan data, bukan evaluasi model.

| Keadaan selang | akurasi |
|---|---|
| selang lurus | 0.834 |
| selang tertekuk 90 derajat | 0.994 |
| ujung selang dinaikkan 30 cm | 0.814 |

## Temuan

- **Sensor arus pompa tidak terpasang.** `daya.csv` hanya berisi kanal kipas; arus kipas median -0.1 mA (kipas mati sepanjang sesi). Satu-satunya sinyal listrik yang terkait pompa adalah tegangan rel 5 V (`bus_mV`) yang turun saat pompa menarik arus.
- Efek keadaan selang pada tegangan rel (relatif terhadap selang lurus, per daya) bertanda sama dan > 10 mV di kedua ulangan blok hanya pada 0/8 pasangan; tegangan rel selang lurus sendiri bergeser +65, +53, +62, +4 mV antar-ulangan blok (per daya). Pergeseran antarblok sebesar atau lebih besar dari efek beban.
- Sidik blok kuat: dua blok berlabel sama dibedakan dengan akurasi 0.81–0.99. Validasi di dalam blok latih akan tampak bagus walau model hanya mengenali blok.
- Wi-Fi ESP32 putus 12 kali: 313 jendela ditolak karena celah sampel, 12 kondisi dikeluarkan karena daya pompa tidak sesuai label.
