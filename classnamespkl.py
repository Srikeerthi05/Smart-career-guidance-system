
import pickle

with open('class_names.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)