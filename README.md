# Dinamika Pertumbuhan Tanaman Hidroponik – Pemodelan Sistem Dinamik

Tugas **TA-06 · Modul Praktikum 6 (Dynamic System Modelling) · Pemodelan dan Simulasi**.
Simulasi Python untuk pertumbuhan tanaman selada pada sistem hidroponik NFT (reservoir 20 L) dengan dua *stock*:
biomassa kering `B(t)` dan konsentrasi nutrisi `N(t)`, dikendalikan oleh laju dosis nutrisi `u(t)`.

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/hidroponik-ta06/blob/main/Hidroponik_TA06.ipynb)

> Ganti `USERNAME` dengan username GitHub Anda (di README ini dan di laporan).

## Model

```
mu(N)  = mu_max * N / (Ks + N)                    # kinetika Monod
dB/dt  = mu(N) * B * (1 - B/B_max)                # pertumbuhan logistik terbatas nutrisi
dN/dt  = u(t)/V - (1/(Y*V)) * dB/dt - kd * N      # neraca massa nutrisi
```

- Jenis model: **ODE** terkopel (reservoir tercampur sempurna, tanpa ketergantungan spasial); **DOF = 0**.
- Steady state: `B* = B_max = 50 g`, `N* = u0/(kd*V) = 75 mg/L`; nilai eigen Jacobian `-0.21` dan `-0.02` (stabil).
- Metode numerik: Euler maju, `dt = 1 hari` (validasi konvergensi terhadap `dt = 0.01`).
- Parameter bersifat **sintetis** (lihat `Params` pada kode) dan dapat dikalibrasi dengan data eksperimen.

## Isi repositori

```
hidroponik-ta06/
├── Hidroponik_TA06.ipynb        # notebook Google Colab (sudah berisi hasil eksekusi)
├── hidroponik_simulasi.py       # kode Python identik dengan notebook (jalankan: python hidroponik_simulasi.py)
├── requirements.txt
├── tests/test_model.py          # 10 uji otomatis (pytest)
├── figures/                     # 9 grafik PNG hasil simulasi
├── output/ringkasan_skenario.csv
├── docs/naskah_video_tutorial.md
├── LICENSE
└── README.md
```

## Cara menjalankan

**Google Colab:** klik badge di atas (atau *File → Open notebook → GitHub*), lalu *Runtime → Run all*.

**Lokal:**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python hidroponik_simulasi.py        # menghasilkan figures/ dan output/ringkasan_skenario.csv
pytest -q                            # menjalankan pengujian
```

## Skenario simulasi (60 hari)

| Skenario | Input u(t) |
|---|---|
| Baseline | u = 30 mg/hari konstan |
| Step | 30 → 60 mg/hari pada t = 20 |
| Impulse | dosis kejut 200 mg pada t = 20 |
| Ramp | naik 1 mg/hari per hari mulai t = 20 |
| Sinusoidal | 30 + 15·sin(2πt/10) |

## Ringkasan hasil

| Skenario | B(60) g | t B50 (hari) | t B95 (hari) | N min | N maks | N(60) | Dosis total (mg) |
|---|---|---|---|---|---|---|---|
| Baseline (u=30) | 49.87 | 25 | 43 | 35.2 | 80.0 | 45.3 | 1800 |
| Tanpa dosis (u=0) | 37.26 | 29 | - | 0.6 | 80.0 | 0.6 | 0 |
| Step | 49.98 | 24 | 40 | 52.2 | 87.3 | 87.3 | 3000 |
| Impulse | 49.92 | 24 | 41 | 41.4 | 80.0 | 50.2 | 2000 |
| Ramp | 49.94 | 24 | 42 | 40.2 | 80.0 | 76.1 | 2580 |
| Sinusoidal | 49.88 | 24 | 43 | 34.7 | 80.8 | 44.6 | 1800 |

Grafik lengkap ada di folder `figures/`; penjelasan dan analisis ada di laporan.

## Mengunggah ke GitHub

```bash
cd hidroponik-ta06
git init
git add .
git commit -m "TA-06: simulasi dinamika pertumbuhan tanaman hidroponik"
git branch -M main
git remote add origin https://github.com/USERNAME/hidroponik-ta06.git
git push -u origin main
```

## Referensi singkat

Andhika (2019); Siradjuddin (n.d.); Sterman (2000); Monod (1949); Resh (2013). Daftar lengkap ada di laporan.
