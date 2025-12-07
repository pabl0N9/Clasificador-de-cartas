from __future__ import annotations

from pathlib import Path
import re
import json

from flask import Flask, jsonify, render_template_string, request
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "modelo_emociones.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        return None, None, "Modelo no encontrado. Ejecuta 'python src/models/train.py' primero."
    modelo = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    return modelo, reverse_encoder, None


app = Flask(__name__)
modelo, reverse_encoder, model_error = load_model()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analizador de Emociones en Cartas</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #111827;
      --accent: #22c55e;
      --muted: #94a3b8;
      --text: #e2e8f0;
      --danger: #ef4444;
      --info: #3b82f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 15% 20%, rgba(34,197,94,0.08), transparent 25%),
                  radial-gradient(circle at 85% 10%, rgba(59,130,246,0.08), transparent 25%),
                  radial-gradient(circle at 50% 80%, rgba(14,165,233,0.08), transparent 30%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .container {
      width: 100%;
      max-width: 960px;
      background: var(--card);
      border: 1px solid rgba(148,163,184,0.12);
      box-shadow: 0 20px 50px rgba(0,0,0,0.35);
      border-radius: 18px;
      padding: 28px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      background: rgba(34,197,94,0.15);
      color: var(--accent);
      padding: 6px 10px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 13px;
      letter-spacing: 0.2px;
    }
    textarea {
      width: 100%;
      min-height: 180px;
      border-radius: 12px;
      border: 1px solid rgba(148,163,184,0.2);
      background: rgba(255,255,255,0.02);
      color: var(--text);
      padding: 14px;
      font-size: 15px;
      line-height: 1.5;
      resize: vertical;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(34,197,94,0.15);
    }
    button {
      background: var(--accent);
      color: #0b1220;
      border: none;
      border-radius: 12px;
      padding: 12px 16px;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
      transition: transform 0.1s ease, box-shadow 0.2s ease, background 0.2s ease;
      width: 100%;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 10px 25px rgba(34,197,94,0.25); }
    button:active { transform: translateY(0); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .card {
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(148,163,184,0.12);
      border-radius: 14px;
      padding: 14px;
    }
    .label { color: var(--muted); font-size: 14px; margin-bottom: 6px; }
    .value { font-size: 20px; font-weight: 700; }
    .progress {
      width: 100%; height: 12px;
      background: rgba(148,163,184,0.15);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress .bar {
      height: 100%;
      background: linear-gradient(90deg, var(--info), var(--accent));
      width: 0%;
      transition: width 0.3s ease;
    }
    .prob-list { list-style: none; padding: 0; margin: 10px 0 0 0; }
    .prob-list li { margin: 4px 0; color: var(--muted); }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .chip {
      background: rgba(148,163,184,0.12);
      border: 1px solid rgba(148,163,184,0.2);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
      transition: border 0.2s ease, transform 0.1s ease;
    }
    .chip:hover { border-color: var(--accent); transform: translateY(-1px); }
    .error {
      background: rgba(239,68,68,0.12);
      border: 1px solid rgba(239,68,68,0.3);
      color: #fecdd3;
      padding: 12px;
      border-radius: 12px;
      margin-top: 12px;
    }
    @media (max-width: 640px) {
      .container { padding: 18px; }
      header { gap: 8px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title">
        <div>
          <div style="font-size: 22px; font-weight: 800;">Analizador de Emociones en Cartas</div>
          <div style="color: var(--muted); font-size: 14px;">Clasifica en enamoramiento, ruptura o confusión</div>
        </div>
      </div>
      <div class="badge">Modelo local</div>
    </header>

    {% if load_error %}
      <div class="error">{{ load_error }}</div>
    {% endif %}

    <form method="POST" action="/" style="display: flex; flex-direction: column; gap: 12px;">
      <label for="texto" class="label">Pega tu carta</label>
      <textarea name="texto" id="texto" placeholder="Escribe o pega el texto para analizar..." required>{{ texto or "" }}</textarea>
      <button type="submit">Analizar emoción</button>
    </form>

    <div class="chips">
      <div class="chip" onclick="setSample('Te amo más cada día y no imagino mi vida sin ti. Eres mi alegría.');">Enamoramiento</div>
      <div class="chip" onclick="setSample('Necesito que hablemos, esto ya no está funcionando entre nosotros. Lo mejor es terminar.');">Ruptura</div>
      <div class="chip" onclick="setSample('No sé qué siento, a veces te extraño y otras necesito espacio. Estoy confundido.');">Confusión</div>
    </div>

    {% if resultado %}
      <div class="grid">
        <div class="card">
          <div class="label">Emoción</div>
          <div class="value">{{ resultado.emocion|upper }}</div>
          <div class="label" style="margin-top: 8px;">Confianza</div>
          <div class="value">{{ '{:.2%}'.format(resultado.confianza) }}</div>
          <div class="progress">
            <div class="bar" style="width: {{ resultado.confianza * 100 }}%;"></div>
          </div>
        </div>
        <div class="card">
          <div class="label">Probabilidades por emoción</div>
          <ul class="prob-list">
            {% for nombre, prob in resultado.detalles.items() %}
              <li><strong>{{ nombre }}:</strong> {{ '{:.2%}'.format(prob) }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>
    {% endif %}
  </div>

  <script>
    function setSample(text) {
      document.getElementById('texto').value = text;
    }
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    global modelo, reverse_encoder, model_error
    texto = ""
    resultado = None
    load_error = None

    if request.method == "POST":
        texto = request.form.get("texto", "").strip()
        if not texto:
            load_error = "Ingresa un texto para analizar."
        elif not modelo:
            load_error = model_error or "Modelo no cargado. Ejecuta 'python src/models/train.py' primero."
        else:
            texto_limpio = preprocess_text(texto)
            probabilidades = modelo.predict_proba([texto_limpio])[0]
            prediccion = modelo.predict([texto_limpio])[0]
            resultado = {
                "emocion": reverse_encoder[prediccion],
                "confianza": float(probabilidades[prediccion]),
                "detalles": {reverse_encoder[i]: float(prob) for i, prob in enumerate(probabilidades)},
            }
            load_error = None

    return render_template_string(HTML_TEMPLATE, texto=texto, resultado=resultado, load_error=load_error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not modelo:
        return jsonify({"error": "Modelo no cargado. Ejecuta 'python src/models/train.py' primero."}), 400

    payload = request.get_json(silent=True) or {}
    texto = payload.get("texto", "")
    if not texto:
        return jsonify({"error": "Se requiere 'texto' en el cuerpo JSON."}), 400

    texto_limpio = preprocess_text(texto)
    probabilidades = modelo.predict_proba([texto_limpio])[0]
    prediccion = modelo.predict([texto_limpio])[0]

    return jsonify(
        {
            "emocion": reverse_encoder[prediccion],
            "confianza": float(probabilidades[prediccion]),
            "detalles": {reverse_encoder[i]: float(prob) for i, prob in enumerate(probabilidades)},
        }
    )


if __name__ == "__main__":
    # Uso simple para desarrollo local
    app.run(host="0.0.0.0", port=8000, debug=True)
