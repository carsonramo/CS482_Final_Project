import pickle
import numpy as np

# ['Albuquerque, NM', 'Phoenix, AZ', 'ABQ', 'PHX', 'WN', 3, 328, 398]
int_columns = ['quarter', 'nsmiles', 'passengers']
string_columns = ['city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']

startingCity = input("Enter starting city (city, ST): ")
endCity = input("Enter destination city (city, ST): ")
startAirport = input("Enter 3 Letter airport code for departure: ")
endAirport = input("Enter 3 Letter airport code for arrival: ")
airline = input("Enter 2 Letter airline code for desired airline: ")
month = input("Enter month as 2 digit number for desired flight time (i.e 04 = April): ")
miles = input("Enter distance in miles between the two airports: ")

month_options = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
while month not in month_options:
    print("invalid month, try again")
    month = input("Enter month as 2 digit number for desired flight time (i.e 04 = April): ")

if month == month_options[0] or month == month_options[1] or month == month_options[2]:
    quarter = 1
elif month == month_options[3] or month == month_options[4] or month == month_options[5]:
    quarter = 2
elif month == month_options[6] or month == month_options[7] or month == month_options[8]:
    quarter = 3
else:
    quarter = 4
features = [startingCity, endCity, startAirport, endAirport, airline, int(quarter), int(miles), 162]

def prediction(model, X):
    with open("OneHotEncoder", "rb") as files:
        encoder = pickle.load(files)
    
    string_X = np.array(X[:len(string_columns)]).reshape(1, -1)
    int_X = np.array(X[len(string_columns):len(string_columns) + len(int_columns)]).reshape(1, -1)
    string_X_encoded = encoder.transform(string_X)

    X_encoded = np.concatenate([string_X_encoded, int_X], axis=1)
    return model.predict(X_encoded)

with open("LinearModel", "rb") as files:
    linearModel = pickle.load(files)

price_prediction = prediction(linearModel, features)
print("\n\n\nFlight price is predicted to cost $", price_prediction[0][0])

with open("RFModel", "rb") as files:
    model = pickle.load(files)

classifer_prediction = prediction(model, features)

if classifer_prediction == [0]:
    price = "Low"
elif classifer_prediction == [1]:
    price = "Medium"
else:
    price = "High"

print("Flight price is estimated to be", price, "cost")