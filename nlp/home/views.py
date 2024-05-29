from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse as HttpJsonResponse
import json
from .models import Works
import datetime


# Create your views here.
def home(request):
    return render(request, "home/home.html")


def get_model(request):
    return render(request, "")


@csrf_exempt
def get_input(request):
    if request.method == "POST" or request.method == "GET":
        data = json.loads(request.body.decode("utf-8"))
        time_save = datetime.datetime.now()
        data_save = Works.objects.create(
            text=data["text"], text_summary=data["text"], data_save=time_save
        )
        data_save.save()
        print(data)
        return HttpJsonResponse(data)


@csrf_exempt
def get_history(request):
    if request.method == "POST" or request.method == "GET":
        data = Works.objects.all()
        data = list(data.values())
        data = data[::-1]
        print(data)
        return render(request, "history/history.html", {"data": data})
