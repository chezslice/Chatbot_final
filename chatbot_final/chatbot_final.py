import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pickle
import numpy
import tflearn
import tensorflow
import random


import json

# Opening provided json file and assigning the data variable to the file.

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

# Creating list variables to assign to words, labels, and docs variables for x and y.

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

# For loops to internate thru the intents and patterns from the input.
# Append words to their apporiate variables.

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
    
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Stemming the words found from the chat to find root word.
# Sorting the words and assigning labels to the sorted list.

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

# Creating training and output lists. For future use.

    training = []
    output = []

# Empty output variable is then internated thru the length of the labels.

    out_empty = [0 for _ in range(len(labels))]

# For x , doc enumerate thru the docs_{x variable.
# Creating a list for bagged the words. And stemming them. And internating the words thru the bag.

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

    # Output variables are then assigned to the out_empty variables and assigning apporiate variables to them.

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

    # Bag variable is then appeneded to training and output data.

        training.append(bag)
        output.append(output_row)

# Using the numpy libary on training and output data.

    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

# Tflearn is the architecture of the neural network being used on the training data 
# 2 hidden layers of neural layers followed by one input layer and one output layer.

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Fitting the model with the training and output data and training the A.I. model 1000 times
# Saving the data to a model data file. With a final result.

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Defining the bag of words and chat variables and making certain predictions on the data.

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()