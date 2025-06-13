import pickle

try:
    with open('scaler1.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)
except Exception as e:
    print(f"Error loading pickle: {e}")
