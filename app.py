import streamlit as st
from openai import OpenAI
import requests
from io import BytesIO
from pydub import AudioSegment

# ==== API Keys ====
# OpenAI Key de prueba temporal
client = OpenAI(api_key="sk-********************")  # Reemplazada aquí por seguridad

# === UI ===
st.title(" GPT con tu voz clonada 🎙️")
st.write("Haz una pregunta y escucha la respuesta con tu voz (gracias a ElevenLabs)")

eleven_api_key = st.text_input(" ElevenLabs API Key", type="password")
voice_id = st.text_input(" Voice ID de ElevenLabs")
user_input = st.text_area("Escribe tu pregunta aquí")

if st.button("Responder con mi voz"):
    if not all([eleven_api_key, voice_id, user_input]):
        st.warning("Por favor completa todos los campos.")
    else:
        # === GPT ===
        with st.spinner("GPT está pensando..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Responde de manera natural y conversacional"},
                    {"role": "user", "content": user_input}
                ]
            )
            texto_respuesta = response.choices[0].message.content
            st.success(" GPT respondió:")
            st.write(texto_respuesta)

        # === ElevenLabs ===
        with st.spinner(" Generando voz con ElevenLabs..."):
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": eleven_api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": texto_respuesta,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.8
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                audio = AudioSegment.from_mp3(audio_data)
                st.audio(audio_data, format="audio/mp3")
            else:
                st.error(" Error al generar el audio. Verifica la API Key o el Voice ID.")


