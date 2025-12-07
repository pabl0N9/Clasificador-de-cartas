from pathlib import Path

import pandas as pd

from src.data.prepare_data import generar_datos_ejemplo
from src.models.train import ModeloEmocionesCartas


def test_model_trains_and_predicts(tmp_path):
    # Prepara un dataset balanceado de ejemplo
    df = generar_datos_ejemplo(n=90, seed=123)
    csv_path = Path(tmp_path) / "cartas_dataset.csv"
    df.to_csv(csv_path, index=False)

    modelo = ModeloEmocionesCartas()
    X, y = modelo.cargar_datos(csv_path=csv_path)

    X_test, y_test, y_pred = modelo.entrenar_modelo(X, y)
    assert len(X_test) == len(y_pred) > 0
    assert set(y_pred).issubset(set(modelo.label_encoder.values()))

    resultado = modelo.predecir("Te amo mas que a nada en este mundo")
    assert resultado["emocion"] in modelo.label_encoder
    assert 0.0 <= resultado["confianza"] <= 1.0
    assert set(resultado["detalles"]).issuperset(set(modelo.label_encoder.keys()))
