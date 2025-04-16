import streamlit as st
import openai
import requests
from io import BytesIO
from pydub import AudioSegment

# === Configuración de claves ===
st.title("GPT con tu voz clonada")
st.write("Haz una pregunta y escucha la respuesta con tu voz (gracias a ElevenLabs)")

openai_api_key = st.text_input("OpenAI API Key", type="password")
eleven_api_key = st.text_input("ElevenLabs API Key", type="password")
voice_id = st.text_input("Voice ID de ElevenLabs")

user_input = st.text_area("Escribe tu pregunta aquí")

if st.button("Responder con mi voz"):
    if not all([openai_api_key, eleven_api_key, voice_id, user_input]):
        st.warning("Por favor completa todos los campos.")
    else:
        # === GPT genera respuesta ===
        openai.api_key = openai_api_key
        with st.spinner("Pensando..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Responde de manera natural y conversacional"},
                    {"role": "user", "content": user_input}
                ]
            )
            texto_respuesta = response['choices'][0]['message']['content']
            st.success("GPT respondió:")
            st.write(texto_respuesta)

        # === ElevenLabs convierte texto a voz ===
        with st.spinner("Generando audio con tu voz..."):
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
                st.error("Error al generar el audio. Revisa tu Voice ID y tu clave.")
