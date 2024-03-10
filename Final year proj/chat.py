# import random
# import json

# import torch

# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "PsycheGuide"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = ""
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")
import random
import json
import torch 
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random

bot_name = "PsycheGuide"
context = dict()
#device= torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = 'cpu'

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size,hidden_size,output_size).to(device)

model.load_state_dict(model_state)
model.eval()



def classify(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output,dim =1)
   
    probs = torch.softmax(output,dim =1)
    prob = probs[0][predicted.item()]
    probs = probs.detach().numpy()
    #print(probs.detach().numpy())
    #print(probs[0][predicted.item()])
    #print(predicted.item())
    #print(tags)
    tag_with_prob = [(tags[i],j) for i,j in enumerate(probs[0])]
    tag_with_prob.sort(key = lambda x :x[1], reverse =True)
    #print(fin)
    return tag_with_prob
def response(sentence):
    results = classify(sentence)
    #print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        #while results and results[0][1]>0.1:
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        context["current"] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if (not 'context_filter' in i and results[0][1]>0.3) or ('context_filter' in i and i['context_filter'] == context["current"] and results[0][1]>0.1) :
                        # a random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)
        return "Sorry! I do not understand!"


if __name__ == "__main__":
   
    print("Lets chat! type 'quit' to exit")
    while True:
        i = input("You: ")
        if i == "quit": 
            break
        print(bot_name," : ",response(i))