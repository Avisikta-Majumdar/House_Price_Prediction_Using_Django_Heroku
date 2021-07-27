from django.shortcuts import render
from . import AllCityColumnName
import numpy as np
import warnings
import joblib
warnings.filterwarnings("ignore")
# Create your views here.
from .models import city
val = None

def homie(request):
    city1 = city()
    city1.name = "Kolkata"
    city1.id = 1

    city2 = city()
    city2.name = "Mumbai"
    city2.id = 2

    city3 = city()
    city3.name = "Delhi"
    city3.id = 3


    showcity = [city1, city2, city3]
    return render(request,'frontfirst.html',{'showcity':showcity})

def home(request):

    dict = {
        "Kolkata" : ["Alipore", "Ballygunge", "Bansdroni", "Barisha Purba Para Road", "Behala", "Dum Dum", "Garia", "Howrah", "Kasba", "Keshtopur", "Konnagar", "Madhyamgram", "Madurdaha Hussainpur", "Mukundapur","Narendrapur", "Nayabad", "New Alipore", "New Town", "Rajarhat", "Salt Lake City", "Sodepur", "Sonarpur", "Tangra", "Tiljala", "Tollygunge", "Uttarpara Kotrung"],
        "Delhi" : ["Chandni Chowk","Rajghat","Islampur"],
        "Mumbai" : ["Andheri East", "Andheri West", "Boisar", "Borivali West", "Chembur", "Dahisar", "Dombivali", "Goregaon East", "Goregaon West", "Kalyan West", "Kamothe", "Kandivali East", "Kandivali West", "Kharghar", "Magathane", "Malad East", "Malad West", "Mira Road East", "Mulund West", "Naigaon East", "Nala Sopara", "Panvel", "Powai", "Taloja", "Thane", "Thane West", "Ulwe", "Ville Parle East", "Virar"]
    }

    name1 = request.POST.get("city")
    global val
    def val():
        return name1
    return render(request, 'frontp.html',{'result':dict[name1]})


def resultprint(request):
    cityname = val()
    Location = request.POST["town"]
    BHK = request.POST["BHK"]
    Area = request.POST["Area Type"]
    resultpredict =""
    # Kolkata
    if cityname == "Kolkata":
        with open('Kolkata_model_joblib','rb') as file:
            mp = joblib.load(file)

        loc_index = AllCityColumnName.kolkata.index(Location)
        x = np.zeros(28)
        x[0] = Area
        x[1] = BHK
        if loc_index >= 0:
            x[loc_index] = 1
        resultpredict = mp.predict([x])[0]

    # Mumbai
    if cityname == "Mumbai":
        with open('Mumbai_model_joblib', 'rb') as file:
            mp = joblib.load(file)

        loc_index = AllCityColumnName.mumbai.index(Location)
        x = np.zeros(len(AllCityColumnName.mumbai))
        x[0] = Area
        x[1] = BHK
        if loc_index >= 0:
            x[loc_index] = 1
        resultpredict = mp.predict([x])[0]
    resultpredict = int(resultpredict)
    resultpredict = str(resultpredict)
    print(resultpredict)


    return render(request,'frontp.html',{'resultpredict':resultpredict})