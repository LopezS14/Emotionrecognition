import streamlit as st
from gtts import gTTS
import pygame
import os
import tempfile
from chat_base import predict_class, get_response, intents
import speech_recognition as sr

def speak(text):
    temp_audio_file = None
    try:
        # Crear archivo temporal
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=text, lang='es')
        tts.save(temp_audio_file.name)
        temp_audio_file_path = temp_audio_file.name
        temp_audio_file.close()

        # Inicializar pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()

        # Esperar a que termine de reproducir
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        if temp_audio_file:
            try:
                os.unlink(temp_audio_file_path)
            except PermissionError:
                pass  # Manejo de excepción si el archivo aún está en uso

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="es-ES")
            return text
        except sr.UnknownValueError:
            return "No entendí lo que dijiste"
        except sr.RequestError:
            return "Error al conectarse al servicio de reconocimiento de voz"

# Interfaz de Streamlit
st.title("Asistente emocional")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    if message["role"] == "Bot":
        with st.chat_message("Bot"):
            st.markdown(message["content"])
    else:
        with st.chat_message("user"):
            st.markdown(message["content"])

# Primer mensaje del bot
if st.session_state.first_message:
    initial_message = "Te doy la bienvenida, estoy aqui para ayudarte."
    with st.chat_message("Bot"):
        st.markdown(initial_message)
    st.session_state.messages.append({"role": "Bot", "content": initial_message})
    st.session_state.first_message = False
    speak(initial_message)

# Colocamos el campo de entrada de texto y el botón de micrófono en la misma fila
col1, col2 = st.columns([8, 1])
with col1:
    prompt = st.chat_input("Estoy para ti, ¿qué necesitas?")  # Cuadro de entrada de texto

with col2:
    if st.button("🎤"):  # Botón de micrófono
        prompt = listen()  # Capturar texto por voz

# Si hay un prompt (de texto o voz), procesarlo
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Implementación del algoritmo de la IA
    insts = predict_class(prompt)  # Predecir la clase de la entrada del usuario
    res = get_response(insts, intents, prompt)  # Llamar a get_response con tres parámetros

    with st.chat_message("Bot"):
        st.markdown(res)
    st.session_state.messages.append({"role": "Bot", "content": res})
    speak(res)