import joblib
from scipy.sparse import hstack
import gradio as gr

import os
import joblib

# Ruta absoluta basada en la ubicación del script
dir_path = os.path.dirname(os.path.abspath(__file__))

import joblib
from scipy.sparse import hstack
import gradio as gr

import os
dir_path = os.path.dirname(os.path.abspath(__file__))

clf = joblib.load(os.path.join(dir_path, "../modelo_debug.joblib"))
tfidf = joblib.load(os.path.join(dir_path, "../vectorizer_debug.joblib"))
scaler = joblib.load(os.path.join(dir_path, "../scaler_debug.joblib"))




# Mapeo de etiquetas
cefr_map = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1"}

def predict_cefr_gradio(text_input):
    if not text_input.strip():
        return "Por favor, introduce un texto."

    # TF-IDF
    X_text = tfidf.transform([text_input])

    # Features numéricas
    wordcount = len(text_input.split())
    mtld = len(set(text_input.split()))
    X_num = scaler.transform([[wordcount, mtld]])

    # Combinar en orden correcto
    X_final = hstack([X_text, X_num])

    # Predicción
    pred = clf.predict(X_final)
    return f"Nivel estimado: {cefr_map[int(pred[0])]}"

# Lanzar interfaz Gradio
demo = gr.Interface(
    fn=predict_cefr_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Escribe aquí un texto en inglés..."),
    outputs="text",
    title="Clasificador de Nivel CEFR",
    description="Introduce un texto y el modelo predecirá tu nivel A1–C1.",
    examples=[
        ["Hi! My name is Laura. I from Brazil. I no speak English very good."],
        ["Yesterday I go to the market with my brother. We buy apples and some milk."],
        ["I think that learning English is important because it help you to find a job."],
        ["Although I have been studying English for several years, I still find it difficult to understand native speakers."],
        ["Despite the widespread availability of online resources, mastering a foreign language still requires consistent effort and exposure."]
    ]
)

print("== TEST DIRECTO AL MODELO ==")
test_text = "Despite the widespread availability of online resources, mastering a foreign language still requires consistent effort and exposure."
print("Predicción directa del modelo:", predict_cefr_gradio(test_text))

demo.launch()
