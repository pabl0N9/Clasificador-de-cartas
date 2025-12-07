from pathlib import Path
import pandas as pd
import random

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cartas_dataset.csv"


def generar_datos_ejemplo(n: int = 300, seed: int | None = None) -> pd.DataFrame:
    """Genera datos de ejemplo para entrenamiento."""
    if seed is not None:
        random.seed(seed)
    patrones = {
        "enamoramiento": [
            "te amo con todo mi corazon", "eres lo mejor que me ha pasado",
            "me haces muy feliz", "quiero pasar mi vida contigo",
            "tu sonrisa ilumina mi dia", "te extrano cuando no estas",
            "contigo todo es mejor", "mi amor por ti crece cada dia",
            "eres mi persona favorita", "no imagino mi vida sin ti",
        ],
        "ruptura": [
            "necesito tiempo para mi", "esto ya no funciona",
            "he decidido terminar nuestra relacion", "no somos felices juntos",
            "me duele decir esto pero", "es mejor que cada uno siga su camino",
            "ya no siento lo mismo", "necesitamos separarnos",
            "no podemos seguir asi", "admitamos que esto se acabo",
        ],
        "confusion": [
            "no se que sentir", "estoy confundido acerca de nosotros",
            "a veces pienso una cosa y a veces otra", "no estoy seguro de mis sentimientos",
            "no entiendo lo que pasa entre nosotros", "tengo sentimientos encontrados",
            "parte de mi quiere pero otra parte no", "no se si es amor o costumbre",
            "estoy en una encrucijada emocional", "mis emociones son un caos",
        ],
    }

    datos = []
    etiquetas = []

    for emocion, frases in patrones.items():
        for _ in range(n // len(patrones)):
            frase_base = random.choice(frases)
            prefijos = ["Realmente ", "A veces ", "Ultimamente ", "Creo que ", "Siento que "]
            sufijos = ["...", "!", ".", "?", " entiendes?"]
            texto = random.choice(prefijos) + frase_base + random.choice(sufijos)

            if random.random() > 0.5:
                extensiones = [
                    " y no puedo evitarlo.",
                    " aunque se que deberia pensarlo mas.",
                    " a pesar de todo lo que ha pasado.",
                    " y espero que tu tambien lo sientas.",
                    " pero tengo miedo de decirtelo.",
                ]
                texto += random.choice(extensiones)

            datos.append(texto)
            etiquetas.append(emocion)

    return pd.DataFrame({"texto": datos, "emocion": etiquetas})


def guardar_datos():
    """Genera y guarda los datos en CSV dentro de data/processed."""
    df = generar_datos_ejemplo(300, seed=42)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    print(f"Datos guardados en '{PROCESSED_PATH}'")
    print(f"Distribucion:\n{df['emocion'].value_counts()}")

    print("\nEjemplos de cada clase:")
    for emocion in df["emocion"].unique():
        ejemplo = df[df["emocion"] == emocion].iloc[0]["texto"]
        print(f"{emocion}: {ejemplo}")


if __name__ == "__main__":
    guardar_datos()
