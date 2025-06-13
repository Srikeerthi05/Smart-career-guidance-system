import pickle

with open('your_file.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
