import pickle

with open('label_encoder.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
