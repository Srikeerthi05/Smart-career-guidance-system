import pickle

with open('label_encoder', 'rb') as f:
    data = pickle.load(f)

print(data)
