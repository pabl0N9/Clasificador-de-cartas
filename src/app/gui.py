from pathlib import Path
import re
import joblib
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "modelo_emociones.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


class InterfazAnalizadorEmociones:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizador de Emociones en Cartas")
        self.root.geometry("600x500")

        try:
            self.modelo = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.reverse_encoder = {v: k for k, v in self.label_encoder.items()}
            self.modelo_cargado = True
        except FileNotFoundError:
            self.modelo_cargado = False
            messagebox.showwarning(
                "Advertencia",
                "Modelo no encontrado. Ejecuta primero 'src/models/train.py' para entrenar y guardar.",
            )

        self.crear_interfaz()

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[\W_]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        titulo = ttk.Label(
            main_frame,
            text="Analizador de Emociones en Cartas",
            font=("Arial", 16, "bold"),
        )
        titulo.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        instrucciones = ttk.Label(
            main_frame,
            text="Escribe o pega una carta para analizar la emocion predominante:",
            font=("Arial", 10),
        )
        instrucciones.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        self.texto_entrada = scrolledtext.ScrolledText(
            main_frame, width=70, height=10, font=("Arial", 10), wrap=tk.WORD
        )
        self.texto_entrada.grid(row=2, column=0, columnspan=2, pady=(0, 15))

        self.btn_analizar = ttk.Button(
            main_frame,
            text="Analizar Emocion",
            command=self.analizar_texto,
            state="normal" if self.modelo_cargado else "disabled",
        )
        self.btn_analizar.grid(row=3, column=0, columnspan=2, pady=(0, 20))

        resultados_frame = ttk.LabelFrame(
            main_frame, text="Resultados del Analisis", padding="15"
        )
        resultados_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(
            resultados_frame, text="Emocion predominante:", font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky=tk.W)

        self.label_emocion = ttk.Label(
            resultados_frame, text="---", font=("Arial", 12, "bold"), foreground="blue"
        )
        self.label_emocion.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(
            resultados_frame, text="Confianza:", font=("Arial", 10, "bold")
        ).grid(row=1, column=0, sticky=tk.W, pady=(10, 0))

        self.label_confianza = ttk.Label(resultados_frame, text="---", font=("Arial", 10))
        self.label_confianza.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        self.barra_confianza = ttk.Progressbar(
            resultados_frame, length=200, mode="determinate"
        )
        self.barra_confianza.grid(row=2, column=0, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))

        ttk.Label(
            resultados_frame, text="Probabilidades por emocion:", font=("Arial", 10, "bold")
        ).grid(row=3, column=0, sticky=tk.W, pady=(15, 5))

        self.label_probabilidades = ttk.Label(resultados_frame, text="", font=("Arial", 9))
        self.label_probabilidades.grid(row=4, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(
            main_frame, text="Ejemplos rapidos:", font=("Arial", 10, "bold")
        ).grid(row=5, column=0, sticky=tk.W, pady=(20, 5))

        ejemplos_frame = ttk.Frame(main_frame)
        ejemplos_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ejemplos = [
            ("Enamoramiento", "Te amo mas cada dia y no imagino mi vida sin tu sonrisa..."),
            ("Ruptura", "Necesito que hablemos, esto ya no esta funcionando entre nosotros."),
            ("Confusion", "No se que hacer, tengo sentimientos encontrados sobre nuestra relacion."),
        ]

        for i, (nombre, texto) in enumerate(ejemplos):
            btn = ttk.Button(
                ejemplos_frame, text=nombre, command=lambda t=texto: self.cargar_ejemplo(t), width=15
            )
            btn.grid(row=0, column=i, padx=5)

    def cargar_ejemplo(self, texto: str):
        self.texto_entrada.delete(1.0, tk.END)
        self.texto_entrada.insert(1.0, texto)

    def analizar_texto(self):
        texto = self.texto_entrada.get(1.0, tk.END).strip()

        if not texto:
            messagebox.showwarning("Advertencia", "Por favor, ingresa un texto para analizar.")
            return
        if len(texto.split()) < 3:
            messagebox.showwarning("Advertencia", "El texto es muy corto. Ingresa al menos 3 palabras.")
            return
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "No hay modelo cargado.")
            return

        try:
            texto_limpio = self.preprocess_text(texto)
            probabilidades = self.modelo.predict_proba([texto_limpio])[0]
            prediccion = self.modelo.predict([texto_limpio])[0]

            emocion = self.reverse_encoder[prediccion]
            confianza = probabilidades[prediccion]

            colores = {
                "enamoramiento": "red",
                "ruptura": "blue",
                "confusion": "purple",
            }

            self.label_emocion.config(text=emocion.upper(), foreground=colores.get(emocion, "black"))
            self.label_confianza.config(text=f"{confianza:.2%}")
            self.barra_confianza["value"] = confianza * 100

            prob_texto = ""
            for i, prob in enumerate(probabilidades):
                emocion_nombre = self.reverse_encoder[i]
                prob_texto += f"- {emocion_nombre}: {prob:.2%}\n"

            self.label_probabilidades.config(text=prob_texto)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrio un error al analizar: {str(e)}")


def main():
    root = tk.Tk()
    app = InterfazAnalizadorEmociones(root)
    root.mainloop()


if __name__ == "__main__":
    main()
