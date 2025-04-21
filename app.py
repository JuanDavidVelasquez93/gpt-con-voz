import streamlit as st
from openai import OpenAI
import requests
from io import BytesIO

# === Función para formatear texto con pausas y acento ===
def formatear_texto_para_voz(texto: str, incluir_etiqueta_idioma=True) -> str:
    texto = texto.strip()
    texto = texto.replace('. ', '.\n')
    texto = texto.replace(',', ',\n')
    texto = texto.replace('?', '?\n')
    texto = texto.replace('!', '!\n')
    
    if incluir_etiqueta_idioma:
        texto = "[es] " + texto

    return texto

# === Interfaz Streamlit ===
st.title("GPT con tu voz clonada")
st.write("Haz una pregunta y escucha la respuesta con tu voz (gracias a ElevenLabs)")

openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
eleven_api_key = st.text_input("🎙️ ElevenLabs API Key", type="password")
voice_id = st.text_input("🧬 Voice ID de ElevenLabs")

user_input = st.text_area("✍️ Escribe tu pregunta aquí")

if st.button("Responder con mi voz"):
    if not all([openai_api_key, eleven_api_key, voice_id, user_input]):
        st.warning("Por favor completa todos los campos.")
    else:
        # === GPT genera respuesta ===
        client = OpenAI(api_key=openai_api_key)
        with st.spinner("🤖 GPT está pensando..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Responde de manera natural y conversacional"},
                    {"role": "user", "content": user_input}
                ]
            )
            texto_respuesta = response.choices[0].message.content
            st.success("✅ GPT respondió:")
            st.write(texto_respuesta)

        # === ElevenLabs convierte texto a voz ===
        with st.spinner("🎧 Generando audio con tu voz..."):
            texto_formateado = formatear_texto_para_voz(texto_respuesta)

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": eleven_api_key,
                "Content-Type": "application/json"
            }

            data = {
                "text": texto_formateado,
                "model_id": "eleven_multilingual_v2",  # tu modelo actual
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 1.0
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                st.audio(audio_data, format="audio/mp3")
            else:
                st.error("❌ Error al generar el audio. Revisa tu Voice ID y tu clave.")
