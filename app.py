import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import requests
from io import BytesIO

# === Cargar modelo ===
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# === Leer archivo TXT cargado ===
def cargar_base_conocimiento():
    with open("BaseConocimiento.txt", "r", encoding="utf-8") as file:
        raw = file.read()

    bloques = raw.split("====")
    datos = []

    for bloque in bloques:
        pregunta, chat, correo, redes = "", "", "", ""
        lineas = bloque.strip().splitlines()
        for linea in lineas:
            if linea.startswith("PREGUNTA:"):
                pregunta = linea.replace("PREGUNTA:", "").strip()
            elif linea.startswith("CHAT:"):
                chat = linea.replace("CHAT:", "").strip()
            elif linea.startswith("CORREO:"):
                correo = linea.replace("CORREO:", "").strip()
            elif linea.startswith("REDES:"):
                redes = linea.replace("REDES:", "").strip()
        if pregunta and chat:
            datos.append({"Pregunta": pregunta, "Respuesta": chat})

    return pd.DataFrame(datos)

# === Buscar mejor respuesta ===
def buscar_respuesta(user_input, df):
    preguntas = df["Pregunta"].tolist()
    embeddings_preguntas = model.encode(preguntas, convert_to_tensor=True)
    embedding_input = model.encode(user_input, convert_to_tensor=True)

    similitudes = util.pytorch_cos_sim(embedding_input, embeddings_preguntas)[0]
    idx = torch.argmax(similitudes).item()
    score = similitudes[idx].item()

    if score > 0.7:
        return df.iloc[idx]["Respuesta"]
    else:
        return "Lo siento, no tengo información sobre eso."

# === ElevenLabs API ===
def generar_audio_elevenlabs(texto, voice_id, api_key):
    # Mejora de formato para pausas naturales
    texto_formateado = texto.replace('. ', '.\n')

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
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
        return BytesIO(response.content)
    else:
        return None

# === Streamlit UI ===
st.set_page_config(page_title="Agente entrenado", page_icon="🤖")
st.title("🤖 Agente entrenado con respuestas definidas")

st.markdown("#### Ingresar API Key y Voice ID de ElevenLabs")
eleven_api_key = st.text_input("🔑 ElevenLabs API Key", type="password")
voice_id = st.text_input("🗣️ Voice ID de tu voz clonada")

user_input = st.text_input("✍️ Escribe tu pregunta:")

if user_input:
    df = cargar_base_conocimiento()
    respuesta = buscar_respuesta(user_input, df)
    st.markdown(f"**Respuesta:** {respuesta}")

    if respuesta != "Lo siento, no tengo información sobre eso." and eleven_api_key and voice_id:
        audio = generar_audio_elevenlabs(respuesta, voice_id, eleven_api_key)
        if audio:
            st.audio(audio, format="audio/mp3")
            st.download_button(
                label="📥 Descargar audio",
                data=audio,
                file_name="respuesta.mp3",
                mime="audio/mpeg"
            )
        else:
            st.warning("No se pudo generar el audio con ElevenLabs.")
