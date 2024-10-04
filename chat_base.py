import random
import io
import streamlit as st
import json
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import subprocess
import pywhatkit as kit  # Importar pywhatkit

lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Función para buscar en Wikipedia
def buscar_en_wikipedia(query):
    try:
        # Realiza la búsqueda en Wikipedia y devuelve el resumen
        respuesta = kit.info(query, 1)
        return respuesta
    except Exception as e:
        return "Esta opcion sigue en proceso de entrenamiento."

# Abrir otro proceso de Python
def abrir_script():
    script_path = "C:/Users/Deyanira LS/Desktop/Asistente-Inteligente-para-el-reconocimiento-de-emociones/reconocimientoEmociones.py"
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, check=True
        )
        return result.stdout, result.stderr
    except FileNotFoundError:
        return "El archivo no fue encontrado.", ""
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr
    except Exception as e:
        return "", str(e)

# Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria o realizamos acciones según la categoría
def get_response(tag, intents_json, last_user_query=None):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    
    if tag == "analisis":
        abrir_script()
     # Si el tag es "buscar_informacion", procesamos la búsqueda
    if tag == "buscar_informacion":
        if last_user_query:
            # Ejecutar búsqueda en Wikipedia si ya tenemos la consulta del usuario
            respuesta_wikipedia = buscar_en_wikipedia(last_user_query)
            result = respuesta_wikipedia
        else:
            # Si no hay una consulta, pedimos al usuario que nos diga sobre qué quiere buscar
            result = "¿Qué tema te gustaría buscar?"
    
    return result
