"""Tes protokol tim 24: blok uji terpisah, bentuk masukan, uji label acak (dilewati bila data belum ada)."""
import numpy as np
import pytest

from src.protokol import data

nyata = pytest.mark.skipif(not data.FOLDER.exists(), reason=f"data/{data.SESI}/ belum ada")


def test_kelas_dan_kanal():
    assert len(data.KELAS) == 3 and data.KANAL == ["bus_mV"]


@nyata
def test_blok_uji_terpisah_dan_semua_kelas_ada():
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    assert Xtr.shape[1:] == (1, int(data.LAJU * data.JENDELA_S))
    b = data.bagian(); c = b["c"]
    assert not set(c["satuan"][b["uji"]].tolist()) & set(c["satuan"][b["latih"] | b["val"]].tolist())
    for y in (ytr, yva, yte):
        assert set(np.unique(y).tolist()) == {0, 1, 2}


@nyata
def test_label_acak_per_kondisi_setara_tebakan():
    from sklearn.ensemble import RandomForestClassifier
    from src.protokol import deret

    b = data.bagian(); c = b["c"]
    F = deret.fitur_statistik(c["X"], data.LAJU)
    kond = c["kondisi"][b["latih"]]
    uk, idx = np.unique(kond, return_index=True)
    peta = dict(zip(uk, np.random.default_rng(0).permutation(c["y"][b["latih"]][idx])))
    m = RandomForestClassifier(200, random_state=0, n_jobs=1).fit(F[b["latih"]], [peta[k] for k in kond])
    acc = float((m.predict(F[b["val"]]) == c["y"][b["val"]]).mean())
    assert acc < 0.6, f"akurasi {acc:.2f} dengan label acak per kondisi"
