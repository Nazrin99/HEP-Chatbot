import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import HepChatbotDataset
from utils import bag_of_words, tokenize, lemmatize
from model import NeuralNetwork
from nltk.corpus import stopwords
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('HEPintents.json', 'r') as data:
    intents = json.load(data)

all_words = []
tags = []
tokenized_word_tag_pair = []
stop_words = set(stopwords.words('english'))

for intent in intents["intents"]:
    tag = intent["tag"]

    tags.append(tag)

    for pattern in intent["patterns"]:

        pattern = pattern.lower()
        word_tokens = tokenize(pattern)
        filtered_words = [word for word in word_tokens if word not in stop_words]
        all_words.extend(filtered_words)
        tokenized_word_tag_pair.append((filtered_words,tag))

ignore_words = ["?", ".", "!", ","]
all_words = [lemmatize(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in tokenized_word_tag_pair:

    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_dim = len(x_train[0])
fc_dim = 16
output_dim = len(tags)

dataset = HepChatbotDataset(x_train, y_train)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

model = NeuralNetwork(input_dim, fc_dim, output_dim).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.15f}")

print(f"Finished Training. Loss : {loss.item():.15f}")

properties = {
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "fc_dim": fc_dim,
    "output_dim": output_dim,
    "all_words": all_words,
    "tags": tags
}

file_name = "model.pth"

torch.save(properties, file_name)

print(f"Training complete. Model saved to {file_name}")