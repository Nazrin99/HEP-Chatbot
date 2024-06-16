from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import numpy as np

lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def lemmatize(word: str):
    return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0

    return bag
