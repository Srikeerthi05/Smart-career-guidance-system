import pickle

with open('scaler.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
