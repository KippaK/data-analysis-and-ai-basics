import pandas as pd
from tensorflow.keras.models import load_model #load model
import pickle #load encoder

# load model
model = load_model('housing-ann-model.keras')

# load encoder
with open('housing-ann-ct.pickle', 'rb') as f:
    ct = pickle.load(f)
    
# load scalers
with open('housing-ann-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)

with open('housing-ann-scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

# ennusta uudella datalla
Xnew = pd.read_csv('housing-new.csv')
Xnew_org = Xnew
Xnew = ct.transform(Xnew)
Xnew = scaler_x.transform(Xnew)
ynew = model.predict(Xnew) 
ynew = scaler_y.inverse_transform(ynew)

# get scaled value back to unscaled
Xnew = scaler_x.inverse_transform(Xnew)

ynew = pd.DataFrame(ynew).reindex()
ynew.columns = ['predicted_price']
df_results = Xnew_org.join(ynew)

# tallennetaan ennusteet csv-tiedostoon
df_results.to_csv('housing-new-ann-with-price.csv', index=False)
