import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
try:
    with open('dataset.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except Exception as e:
    print(f"Error loading resources: {e}")
    exit(1)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of the input sentence
def predicts(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ErrorThreshold = 0.25
    final_results = [[i, r] for i, r in enumerate(res) if r > ErrorThreshold]

    final_results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in final_results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I didn't understand that."

# Main loop for chatbot interaction
print("GO! Bot is running!")

while True:
    info = input("You: ")
    if info.lower() == "exit":
        print("Bot: Goodbye!")
        break
    ints = predicts(info)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
