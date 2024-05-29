from django.contrib import admin
from django.urls import path, include

from . import views

urlpatterns = [
    path("", views.home),
    path("get_input", views.get_input),
    path("get_history", views.get_history),
]
