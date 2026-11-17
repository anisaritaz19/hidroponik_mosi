# Naskah Video Tutorial – Dinamika Pertumbuhan Tanaman Hidroponik

Durasi target: ± 9 menit 50 detik. Ganti [Nama], [NIM], [Kelompok], dan [USERNAME] sebelum merekam.

| Waktu | Layar / Visual | Narasi |
|---|---|---|
| 00:00–00:30 | Layar judul: “Dinamika Pertumbuhan Tanaman Hidroponik – Pemodelan Sistem Dinamik dengan Python”. Tampilkan nama, NIM, kelompok. | Halo, saya [Nama], NIM [NIM], kelompok [Kelompok]. Pada video ini saya akan menjelaskan tugas TA-06 Pemodelan dan Simulasi: memodelkan dinamika pertumbuhan tanaman hidroponik menggunakan Python di Google Colab, lengkap dengan perhitungan manual, steady state, dan simulasi sinyal step, impulse, ramp, serta sinusoidal. |
| 00:30–01:20 | Slide/laporan bagian 3a: deskripsi studi kasus; tampilkan Gambar 1 (diagram stock-flow). | Studi kasusnya adalah sistem hidroponik NFT berisi 12 tanaman selada dengan satu reservoir 20 liter. Pompa dosis menambahkan nutrisi dengan laju u(t). Tanaman menyerap nutrisi untuk tumbuh, dan sebagian nutrisi hilang karena pengendapan. Tujuannya: memprediksi biomassa B dan konsentrasi nutrisi N dari waktu ke waktu, mencari steady state, dan melihat respon sistem terhadap berbagai sinyal dosis. |
| 01:20–02:30 | Sorot diagram: stock N dan B, flow inflow, pertumbuhan, outflow, penyerapan, dan kotak umpan balik. | Pada diagram stock-flow ada dua stock: konsentrasi nutrisi N dan biomassa kering B. Aliran masuk adalah dosis pompa u(t) dibagi volume. Aliran pertumbuhan dihitung dari kinetika Monod dikalikan biomassa dan faktor kapasitas satu dikurangi B per B maksimum. Penyerapan nutrisi mengurangi N, dan kehilangan nutrisi sebanding dengan kd kali N. Ada tiga umpan balik: satu penguat, yaitu makin besar tanaman makin cepat tumbuh, dan dua penyeimbang, yaitu kapasitas sistem dan menipisnya nutrisi. |
| 02:30–03:20 | Tampilkan Tabel 3–4 dan persamaan (laporan bagian 3a–3b). | Karena reservoir dianggap tercampur sempurna, sistem tidak bergantung ruang, sehingga modelnya berupa ODE. Ada tiga variabel tak diketahui, yaitu B, N, dan laju spesifik mu, dengan tiga persamaan, sehingga derajat kebebasannya nol. Input kontrolnya adalah u(t); konstanta berupa parameter model; gangguan berupa kehilangan nutrisi atau dosis kejut. Outputnya adalah state B dan N, dengan N dan B sebagai variabel yang dikontrol. |
| 03:20–04:30 | Tampilkan Tabel 5–6 perhitungan manual; sorot hasil langkah pertama. | Sekarang perhitungan manual dengan metode Euler, langkah satu hari. Pada t sama dengan nol, B 0,5 gram dan N 80 miligram per liter. Laju spesifik mu sama dengan 0,35 kali 80 dibagi 130, hasilnya 0.2154. Laju pertumbuhan dB per dt 0.1066 gram per hari, dan dN per dt -0.2333. Maka biomassa hari pertama 0.6066 gram dan nutrisi 79.7667 miligram per liter. Angka ini nanti harus sama dengan keluaran program. |
| 04:30–05:40 | Buka Google Colab (notebook dari GitHub). Jalankan sel parameter, model, dan sel perhitungan manual. Tunjukkan tabel keluaran. | Kita buka notebook di Google Colab. Sel pertama mengimpor numpy dan matplotlib. Kemudian kelas Params menyimpan semua parameter. Fungsi mu_monod dan turunan menerjemahkan persamaan ke kode, dan fungsi simulasi melakukan loop Euler. Saat sel perhitungan manual dijalankan, tabel yang keluar sama persis dengan hitungan tangan tadi, jadi implementasinya benar. |
| 05:40–06:50 | Jalankan sel steady state; tampilkan Gambar 2 dan 3. | Untuk steady state, kita atur turunan sama dengan nol. Hasilnya B bintang sama dengan B maksimum, 50 gram, dan N bintang sama dengan u nol dibagi kd kali V, yaitu 75 miligram per liter. Nilai eigen Jacobian minus 0,21 dan minus 0,02, keduanya negatif, jadi titik ini stabil. Pada grafik, biomassa berbentuk S menuju 50 gram, sedangkan nutrisi turun dulu sampai sekitar 35 miligram per liter lalu naik pelan menuju 75. Potret fase menunjukkan sembilan kondisi awal berbeda semuanya menuju titik yang sama. |
| 06:50–08:20 | Jalankan sel skenario satu per satu; tampilkan Gambar 4, 5, 6, 7. Sorot panel u(t), N(t), B(t). | Selanjutnya empat sinyal uji. Step: dosis naik dari 30 ke 60 pada hari ke-20; nutrisi turun lebih landai, minimumnya 52 dibanding 35, lalu naik terus. Impulse: dosis kejut 200 miligram menaikkan nutrisi seketika 10 miligram per liter, lalu meluruh. Ramp: dosis naik linear sehingga nutrisi naik melengkung dan belum mencapai steady state dalam 60 hari. Sinusoidal: nutrisi hanya beriak kecil karena reservoir meredam osilasi cepat. Pada semua kasus biomassa akhir hampir sama, sekitar 50 gram. |
| 08:20–09:10 | Tampilkan Gambar 8, tabel ringkasan, dan Gambar 9 (validasi). | Pada perbandingan, tanpa dosis nutrisi habis dan biomassa hanya 37.3 gram. Untuk validasi, galat maksimum Euler dengan langkah satu hari sekitar 3.4 gram, dan turun menjadi 0.3 gram pada langkah 0,1 hari, sesuai sifat orde satu. Neraca massa nutrisi juga terpenuhi dengan residu mendekati nol. |
| 09:10–09:50 | Tampilkan repositori GitHub dan halaman laporan (Link, Referensi). | Kesimpulannya, sistem hidroponik dapat dimodelkan sebagai dua ODE terkopel dengan steady state stabil di 50 gram dan 75 miligram per liter. Kode lengkap, notebook, dan laporan ada di GitHub; tautannya di deskripsi video. Parameter yang dipakai sintetis, jadi untuk penggunaan nyata perlu kalibrasi data eksperimen. Terima kasih telah menonton. |

## Checklist rekaman

- Siapkan Colab dengan Runtime baru dan jalankan Runtime → Run all sekali sebelum merekam agar tidak ada error.
- Zoom browser 110–125% supaya kode terbaca; tutup tab dan notifikasi.
- Rekam layar 1080p (OBS/Zoom/Colab + Loom) dengan mikrofon eksternal; unggah ke YouTube (Tidak publik/Unlisted bila diminta).
- Durasi target 9–10 menit. Bacakan narasi dengan tempo santai; tampilkan wajah/nama di awal bila diwajibkan dosen.
- Salin tautan YouTube ke bagian 4 laporan, README, dan Edlink.

## Judul, deskripsi, dan tag YouTube

```
Judul YouTube: Dinamika Pertumbuhan Tanaman Hidroponik – Pemodelan Sistem Dinamik Python (TA-06 Pemodelan dan Simulasi)
Deskripsi: Tutorial pemodelan dan simulasi pertumbuhan tanaman hidroponik (biomassa dan nutrisi) dengan Python di Google Colab: stock-flow, ODE, perhitungan manual Euler, steady state, dan input step/impulse/ramp/sinusoidal.
GitHub: https://github.com/[USERNAME]/hidroponik-ta06
Colab: https://colab.research.google.com/github/[USERNAME]/hidroponik-ta06/blob/main/Hidroponik_TA06.ipynb
Tag: pemodelan dan simulasi, sistem dinamik, hidroponik, python, google colab, stock flow, metode euler
```
