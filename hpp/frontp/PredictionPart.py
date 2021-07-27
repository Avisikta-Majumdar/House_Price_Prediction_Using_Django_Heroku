#This .py will contain all the method name which will help us to predict the price based on location , area,bhk

import AllCityColumnName
import pickle
import numpy as np
#Kolkata
def predict_Kolkata_house_price(Location , Area, BHK):
    with open('Kolkata_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.kolkata.index(Location)
    x = np.zeros(28)
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]
#Mumbai
def predict_Mumbai_house_price(Location , Area, BHK):
    with open('Mumbai_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.mumbai.index(Location)
    x = np.zeros(len(AllCityColumnName.mumbai))
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]

#Bangalore
def predict_Bangalore_house_price(Location , Area, BHK):
    with open('Bangalore_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.bangalore.index(Location)
    x = np.zeros(len(AllCityColumnName.bangalore))
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]

#Chennai
def predict_Chennai_house_price(Location , Area, BHK):
    with open('Chennai_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.chennai.index(Location)
    x = np.zeros(len(AllCityColumnName.chennai))
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]

#Hyderabad
def predict_Hyderabad_house_price(Location , Area, BHK):
    with open('Hyderabad_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.hyderabad.index(Location)
    x = np.zeros(len(AllCityColumnName.hyderabad))
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]

#Delhi
def predict_Delhi_house_price(Location , Area, BHK):
    with open('Delhi_model_pickle', 'rb') as file:
        mp = pickle.load(file)

    loc_index = AllCityColumnName.delhi.index(Location)
    x = np.zeros(len(AllCityColumnName.delhi))
    x[0] = Area
    x[1] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return mp.predict([x])[0]