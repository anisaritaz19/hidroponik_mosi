"""Uji otomatis model hidroponik. Jalankan: pytest -q"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import hidroponik_simulasi as hs

P = hs.P


def test_steady_state_teori():
    B, N = hs.steady_state(P.u0, P)
    assert B == P.B_max
    assert np.isclose(N, P.u0 / (P.kd * P.V))
    dB, dN = hs.turunan(B, N, P.u0, P)
    assert abs(dB) < 1e-12 and abs(dN) < 1e-12


def test_kestabilan():
    ks = hs.analisis_kestabilan(P)
    assert np.all(ks["eig_hidup"].real < 0)      # titik setimbang tanaman hidup stabil
    assert np.any(ks["eig_B0"].real > 0)         # titik B = 0 tidak stabil


def test_konvergensi_ke_steady_state():
    r = hs.simulasi(hs.buat_sinyal("baseline", P), P, t_end=600)
    B, N = hs.steady_state(P.u0, P)
    assert abs(r["B"][-1] - B) < 1e-3 and abs(r["N"][-1] - N) < 0.1


def test_mulai_di_steady_state_tetap_datar():
    B, N = hs.steady_state(P.u0, P)
    r = hs.simulasi(hs.buat_sinyal("baseline", P), P, t_end=100, B0=B, N0=N)
    assert np.allclose(r["B"], B) and np.allclose(r["N"], N)


def test_neraca_massa_nutrisi():
    for nama in ("baseline", "step", "ramp", "sinus"):
        r = hs.simulasi(hs.buat_sinyal(nama, P), P)
        assert abs(hs.cek_neraca_massa(r, P)) < 1e-8


def test_impulse_luas_sama_dengan_dosis():
    r = hs.simulasi(hs.buat_sinyal("impulse", P), P)
    ekstra = (r["u"] - P.u0)[:-1].sum() * r["dt"]
    assert np.isclose(ekstra, P.impulse_M)


def test_sinyal_step_ramp_sinus():
    assert hs.buat_sinyal("step", P)(19) == P.u0
    assert hs.buat_sinyal("step", P)(20) == P.u0 + P.step_amp
    assert np.isclose(hs.buat_sinyal("ramp", P)(30), P.u0 + 10 * P.ramp_slope)
    assert np.isclose(hs.buat_sinyal("sinus", P)(2.5), P.u0 + P.sin_amp)


def test_batas_fisik():
    for nama in hs.SKENARIO:
        r = hs.simulasi(hs.buat_sinyal(nama, P), P)
        assert r["B"].min() >= 0 and r["B"].max() <= P.B_max + 1e-9
        assert r["N"].min() >= 0


def test_euler_orde_satu_galat_menurun():
    _, _, galat = hs.validasi_langkah_waktu(P)
    gb = [g[1] for g in galat]
    assert gb[0] > gb[1] > gb[2]


def test_tabel_manual_langkah_pertama():
    r = hs.tabel_manual(P, 5)[0]
    assert np.isclose(r["mu"], 0.35 * 80 / 130)
    assert np.isclose(r["dBdt"], r["mu"] * 0.5 * (1 - 0.5 / 50))
    assert np.isclose(r["dNdt"], 30 / 20 - r["dBdt"] / (0.04 * 20) - 0.02 * 80)
