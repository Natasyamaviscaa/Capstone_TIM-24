# Gladi rencana dengan mode simulasi

```bash
cd meja-uji-5v/perekam
sed -e 's/^ulangan: 4/ulangan: 1/' ../rencana/t24-pompa-arus.yaml > /tmp/t24-gladi.yaml
python meja.py --simulasi jalankan /tmp/t24-gladi.yaml
```

Simulasi selalu menulis kanal `pompa` di `daya.csv`; meja nyata tidak. Sebelum slot nyata, periksa kartu *Arus* di
halaman Meja 5 Volt: kartu pompa harus muncul. Di sesi ini tidak muncul, dan sesi tetap dijalankan.
