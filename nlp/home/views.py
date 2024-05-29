from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse as HttpJsonResponse
import json
from .models import Works


# Create your views here.
def home(request):
    return render(request, "home/home.html")


def get_model(request):
    return render(request, "")


@csrf_exempt
def get_input(request):
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        print(data)
        return HttpJsonResponse(data)
