import json
import random

import torch
from model import NeuralNetwork
from utils import bag_of_words, tokenize
from nltk.corpus import stopwords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('HEPintents.json', 'r') as intent_data:
    intents = json.load(intent_data)

def load_model(path: str):
    model = torch.load(path)

    input_dim = model["input_dim"]
    fc_dim = model["fc_dim"]
    output_dim = model["output_dim"]
    all_words = model["all_words"]
    tags = model["tags"]
    model_state = model["model_state"]

    model = NeuralNetwork(input_dim, fc_dim, output_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    return {"all_words": all_words, "tags": tags, "model": model}.values()

def chat(user_input):
    if(len(user_input) < 10):
        response = "Your query is too short. Please input a longer prompt for more contextual answers"
        return response

    all_words, tags, model = load_model("model.pth")
    punctuations = ["?", ".", "!", ","]

    sentence = [word for word in tokenize(user_input.lower()) if word not in stopwords.words('english')]
    filtered_sentence = [word for word in sentence if word not in punctuations]
    x = bag_of_words(filtered_sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    print(probability.item())

    if probability.item() > 0.70:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])

    else:
        response = "Apologies. I do not have enough knowledge or resources to answer that question for now. Please direct your question to the HEP department at hep@um.du.my or by calling 03-79673506"

    return response
