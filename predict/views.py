import sys
import os

# Add ml/ folder to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml'))

from django.shortcuts import render
from ml.predict import predict_price

def index(request):
    return render(request, 'predict/index.html')

def predict_view(request):
    if request.method == 'POST':
        try:
            features = {
                "MedInc"     : float(request.POST.get('MedInc', 0)),
                "HouseAge"   : float(request.POST.get('HouseAge', 0)),
                "AveRooms"   : float(request.POST.get('AveRooms', 0)),
                "AveBedrms"  : float(request.POST.get('AveBedrms', 0)),
                "Population" : float(request.POST.get('Population', 0)),
                "AveOccup"   : float(request.POST.get('AveOccup', 0)),
                "Latitude"   : float(request.POST.get('Latitude', 0)),
                "Longitude"  : float(request.POST.get('Longitude', 0)),
            }
            price = predict_price(features)
            return render(request, 'predict/result.html', {
                'price'   : f"{price:,.2f}",
                'features': features
            })
        except Exception as e:
            return render(request, 'predict/index.html', {'error': str(e)})

    return render(request, 'predict/index.html')