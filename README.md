# Clasificador-de-cartas

IA para leer y clasificar cartas en tres emociones: enamoramiento, ruptura y confusion.

## Requisitos rapidos
- Python 3.11+
- Windows/macOS/Linux
- Opcional pero recomendado: descargar stopwords de NLTK con `python -m nltk.downloader stopwords`. Sin red se usa una lista minima incluida.

## Configuracion
1) Crear entorno: `python -m venv venv`
2) Activar entorno (PowerShell): `.\venv\Scripts\Activate.ps1`
3) Instalar dependencias: `pip install -r requirements.txt`

## Preparar datos y entrenar
- Generar dataset sintetico (si no existe `data/processed/cartas_dataset.csv`):  
  `python src/data/prepare_data.py`
- Entrenar y guardar el modelo/encoder:  
  `python src/models/train.py`  
  Crea `models/modelo_emociones.pkl` y `models/label_encoder.pkl`.

## Ejecutar la IA
- Servidor web (Flask): `python src/app/web.py` y abrir `http://localhost:8000`
- Interfaz de escritorio (Tkinter): `python src/app/gui.py`

## Pruebas
`pytest`
