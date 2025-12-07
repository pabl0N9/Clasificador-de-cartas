import pandas as pd

from src.data.prepare_data import generar_datos_ejemplo


def test_generar_datos_ejemplo_produces_expected_shape():
    df = generar_datos_ejemplo(n=60, seed=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 60
    assert set(df["emocion"].unique()) == {"enamoramiento", "ruptura", "confusion"}
    assert df["texto"].str.len().min() > 0


def test_generar_datos_ejemplo_is_deterministic_with_seed():
    df1 = generar_datos_ejemplo(n=30, seed=123)
    df2 = generar_datos_ejemplo(n=30, seed=123)

    assert df1.equals(df2)
