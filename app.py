from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load model
with open('./Models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
df = pd.read_csv('./Models/x_train.csv')

# Ensure all columns are of the same type
df = df.astype(int)

# Drop unnecessary columns (e.g., index columns)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buy')
def project():
    return render_template('buy.html')

@app.route("/prediction", methods=['POST'])
def prediction():
    status = False
    price = None

    # Get form inputs
    sqft = request.form.get('sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')
    location = request.form.get('location')

    status = True

    # Find the index of the location column in the dataframe
    loc_index = np.where(df.columns == location)[0]

    if loc_index.size == 0:
        raise ValueError("Location not found in the dataset.")

    # Initialize a feature vector with the correct number of features
    x = np.zeros(len(df.columns))

    # Set the first three values as sqft, bath, and bhk
    x[0] = float(sqft)
    x[1] = float(bath)
    x[2] = float(bhk)

    # Set the value of the location feature to 1
    if loc_index.size > 0:
        x[loc_index[0]] = 1

    # Predict the price
    price = model.predict([x])[0] * 100000  # Ensure you're using the correct scaling for the output

    print(price)

    # Render the buy.html template with the predicted price and status
    return render_template('buy.html', price=int(price), status=status)

if __name__ == "__main__":
    app.run(debug=True)
