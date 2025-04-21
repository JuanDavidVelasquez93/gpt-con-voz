import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import requests
from io import BytesIO

# === Cargar modelo de similitud ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Leer base de conocimiento desde .txt ===
def cargar_base_conocimiento():
    with open("BaseConocimiento.txt", "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = contenido.split("====")
    preguntas = []
    respuestas = []

    for bloque in bloques:
        lineas = bloque.strip().splitlines()
        pregunta = ""
        chat = ""
        for linea in lineas:
            if linea.startswith("PREGUNTA:"):
                pregunta = linea.replace("PREGUNTA:", "").strip()
            elif linea.startswith("CHAT:"):
                chat = linea.replace("CHAT:", "").strip()
        if pregunta and chat:
            preguntas.append(pregunta)
            respuestas.append(chat)

    return pd.DataFrame({"Pregunta": preguntas, "Respuesta": respuestas})

# === Buscar mejor coincidencia ===
def buscar_respuesta(user_input, df):
    preguntas = df["Pregunta"].tolist()
    embeddings = model.encode(preguntas, convert_to_tensor=True)
    embedding_input = model.encode(user_input, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(embedding_input, embeddings)[0]

    idx = torch.argmax(similitudes).item()
    score = similitudes[idx].item()

    if score > 0.7:
        return df.iloc[idx]["Respuesta"]
    else:
        return "Lo siento, no tengo información sobre eso."

# === Streamlit UI ===
st.title("🤖 Agente con base de conocimiento")
st.write("Escribe tu pregunta y responderé solo si está entrenada.")

st.markdown("#### Configura ElevenLabs para voz personalizada")
eleven_api_key = st.text_input("🔑 ElevenLabs API Key", type="password")
voice_id = st.text_input("🗣️ Voice ID")

user_input = st.text_area("✍️ Escribe tu pregunta aquí")

if user_input:
    df = cargar_base_conocimiento()
    respuesta = buscar_respuesta(user_input, df)
    st.markdown(f"**Respuesta:** {respuesta}")

    # Generar audio si hay voz configurada
    if respuesta != "Lo siento, no tengo información sobre eso." and eleven_api_key and voice_id:
        with st.spinner("🎙️ Generando audio..."):
            texto_formateado = respuesta.strip().replace('. ', '.\n')
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": eleven_api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": texto_formateado,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 1.0
                }
            }
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                st.audio(audio_data, format="audio/mp3")
                st.download_button("📥 Descargar audio", audio_data, file_name="respuesta.mp3", mime="audio/mpeg")
            else:
                st.error("Error al generar audio con ElevenLabs.")
