
import pickle

with open('scaler1.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)