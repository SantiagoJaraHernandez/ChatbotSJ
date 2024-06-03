import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import tkinter as tk
from tkinter import DISABLED, END, NORMAL, messagebox
from datetime import datetime

lemmatizer = WordNetLemmatizer()

# Cargar los datos y modelos necesarios
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Preprocesamiento de texto más avanzado
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in nltk.corpus.stopwords.words('english')]
    return sentence_words

# Actualizar la función bag_of_words para reflejar el nuevo preprocesamiento
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

# Mejora en la predicción de la clase
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category

# Mejora en la generación de respuestas
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)
    return "Lo siento, no puedo entender eso en este momento."

# Función principal para obtener respuesta
def respuesta(message):
    intent_tag = predict_class(message)
    response = get_response(intent_tag, intents)
    return response if response else "Lo siento, aún no tengo esa información."

# Función para enviar un mensaje
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Tú: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))

        # Si el mensaje comienza con 'nota', se crea una nota
        if msg.lower().startswith('nota'):
            note = create_note(msg)
            ChatBox.insert(END, "OsirisAI: " + note + '\n\n')
        # Si el mensaje es sobre la hora, se muestra la hora actual
        elif 'hora' in msg.lower():
            current_time = read_time()
            ChatBox.insert(END, "OsirisAI: " + current_time + '\n\n')
        else:
            res = respuesta(msg)
            # Si la respuesta es relevante, mostrarla
            if res:
                ChatBox.insert(END, "OsirisAI: " + res + '\n\n')
            else:
                ChatBox.insert(END, "OsirisAI: Lo siento, aún no tengo esa información.\n\n")

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
# Función para crear una nota
def create_note(msg):
    # Extraer el contenido de la nota del mensaje
    note_content = msg[len('nota'):].strip()
    return f"Nota creada: {note_content}"

# Función para leer la hora
def read_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"Son las {current_time}."

# Crear la ventana principal
root = tk.Tk()
root.title("Chatbot")

# Crear la caja de chat
ChatBox = tk.Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)

# Deshabilitar la caja de chat para evitar que el usuario escriba directamente
ChatBox.config(state=DISABLED)

# Agregar una barra de desplazamiento vertical a la caja de chat
scrollbar = tk.Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Crear la caja de entrada para que el usuario ingrese mensajes
EntryBox = tk.Text(root, bd=0, bg="white",width="29", height="5", font="Arial")

# Crear un botón para enviar mensajes
SendButton = tk.Button(root, font=("Verdana",12,'bold'), text="Enviar", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command=send )

# Posicionar los elementos en la ventana principal
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

# Ejecutar la aplicación
root.mainloop()
