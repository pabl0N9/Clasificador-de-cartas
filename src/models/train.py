from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_pipeline
from nltk.corpus import stopwords
import nltk

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "cartas_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo_emociones.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

def load_spanish_stopwords():
    """
    Carga stopwords en espanol.
    Usa NLTK si esta disponible; de lo contrario recurre a una lista minima local
    para evitar fallar en entornos sin red (por ejemplo, CI/tests).
    """
    try:
        nltk.data.find("corpora/stopwords")
        return stopwords.words("spanish")
    except LookupError:
        return [
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
            "las", "por", "un", "para", "con", "no", "una", "su", "al",
            "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este",
            "si", "porque", "esta", "entre", "cuando", "muy", "sin",
            "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
            "desde", "todo", "nos", "durante", "todos", "uno", "les",
            "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
            "e", "esto", "mi", "antes", "algunos", "que", "unos",
            "yo", "otro", "otras", "otra", "el", "tanto", "esa", "estos",
            "mucho", "quienes", "nada", "muchos", "cual", "poco",
            "ella", "estar", "estas", "algunas", "algo", "nosotros",
            "mi", "mis", "tus", "te", "ti", "tu", "ellas", "nosotras",
            "vosostros", "vosostras", "os", "mio", "mia", "mios", "mias",
            "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos",
            "suyas", "nuestro", "nuestra", "nuestros", "nuestras",
            "vuestro", "vuestra", "vuestros", "vuestras", "esos",
            "esas", "estoy", "estas", "esta", "estamos", "estais",
            "estan", "este", "estes", "estemos", "esteis", "esten",
            "estare", "estaras", "estara", "estaremos", "estareis",
            "estaran", "estaria", "estarias", "estariamos", "estariais",
            "estarian", "estaba", "estabas", "estabamos", "estabais",
            "estaban", "estuve", "estuviste", "estuvo", "estuvimos",
            "estuvisteis", "estuvieron", "estuviera", "estuvieras",
            "estuvieramos", "estuvierais", "estuvieran", "estuviese",
            "estuvieses", "estuviesemos", "estuvieseis", "estuviesen",
            "estando", "estado", "estada", "estados", "estadas", "estad",
        ]


class ModeloEmocionesCartas:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = {"enamoramiento": 0, "ruptura": 1, "confusion": 2}
        self.reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        self.stopwords = load_spanish_stopwords()

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[\W_]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def cargar_datos(self, csv_path: Path = DATA_PATH):
        df = pd.read_csv(csv_path)
        df["texto_limpio"] = df["texto"].apply(self.preprocess_text)
        df["etiqueta"] = df["emocion"].map(self.label_encoder)
        return df["texto_limpio"], df["etiqueta"]

    def entrenar_modelo(self, X, y):
        self.pipeline = imbalanced_pipeline(
            TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words=self.stopwords,
            ),
            SMOTE(random_state=42),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=22,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
            ),
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Entrenando modelo...")
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        print("\n=== RESULTADOS DEL MODELO ===")
        print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")
        print("\nReporte de Clasificacion:")
        print(
            classification_report(
                y_test, y_pred, target_names=list(self.label_encoder.keys())
            )
        )

        scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy")
        print(
            f"\nValidacion Cruzada (5-fold): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
        )

        return X_test, y_test, y_pred

    def guardar_modelo(self, model_path: Path = MODEL_PATH):
        if not self.pipeline:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        print(f"\nModelo guardado como '{model_path}'")
        print(f"Encoder guardado como '{ENCODER_PATH}'")

    def cargar_modelo(self, model_path: Path = MODEL_PATH):
        self.pipeline = joblib.load(model_path)
        self.label_encoder = joblib.load(ENCODER_PATH)
        self.reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        print(f"Modelo cargado desde '{model_path}'")

    def predecir(self, texto: str):
        if not self.pipeline:
            raise ValueError("Modelo no entrenado. Entrena o carga un modelo primero.")

        texto_limpio = self.preprocess_text(texto)
        probabilidades = self.pipeline.predict_proba([texto_limpio])[0]
        prediccion = self.pipeline.predict([texto_limpio])[0]

        return {
            "emocion": self.reverse_encoder[prediccion],
            "confianza": probabilidades[prediccion],
            "detalles": {self.reverse_encoder[i]: prob for i, prob in enumerate(probabilidades)},
        }

    def analizar_ejemplos(self, ejemplos):
        print("\n=== ANALISIS DE EJEMPLOS ===")
        for ejemplo in ejemplos:
            resultado = self.predecir(ejemplo)
            print(f"\nTexto: '{ejemplo[:50]}...'")
            print(f"Emocion predicha: {resultado['emocion']}")
            print(f"Confianza: {resultado['confianza']:.2%}")
            print(f"Probabilidades: {resultado['detalles']}")


def main():
    modelo = ModeloEmocionesCartas()
    print(f"Cargando datos desde: {DATA_PATH}")
    X, y = modelo.cargar_datos()
    modelo.entrenar_modelo(X, y)
    modelo.guardar_modelo()

    ejemplos_prueba = [
        "Te amo mas que a nada en este mundo y quiero estar contigo siempre",
        "Creo que debemos terminar esta relacion, ya no somos felices juntos",
        "No se que siento por ti, a veces te amo y a veces necesito espacio",
        "Tu presencia llena mi vida de alegria y no imagino un futuro sin ti",
        "Necesito tiempo para pensar en lo nuestro, estoy muy confundido",
    ]
    modelo.analizar_ejemplos(ejemplos_prueba)
    return modelo


if __name__ == "__main__":
    main()
