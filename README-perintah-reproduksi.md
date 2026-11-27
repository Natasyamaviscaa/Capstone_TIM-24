Proses reproduksi eksperimen dilakukan melalui tahapan berikut:

1. **Persiapan lingkungan**  
   Menyiapkan repository, menginstal dependensi, dan memastikan lingkungan Python sudah siap digunakan.

2. **Memuat data**  
   Memuat data primer meja menggunakan modul protokol data yang telah disediakan dalam repository.

3. **Pemrosesan data**  
   Melakukan pemotongan sinyal menjadi sampel sesuai protokol dan pembagian data train, validation, dan test.

4. **Menjalankan model**  
   Menjalankan masing-masing jalur model anggota melalui file `run.py` pada folder model masing-masing.

5. **Evaluasi model**  
   Mengevaluasi hasil prediksi menggunakan evaluator standar tim dan menyimpan hasil evaluasi pada folder `results/`.

6. **Komparasi hasil**  
   Menjalankan script `run_all.py` untuk menggabungkan hasil eksperimen dan membentuk tabel komparasi model.
