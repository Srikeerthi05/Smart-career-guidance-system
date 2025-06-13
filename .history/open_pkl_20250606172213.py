import pickle

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
