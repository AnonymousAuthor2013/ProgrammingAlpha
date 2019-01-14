from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
# Create your views here.

def mainPage(request:HttpRequest):
    info="Hello to programming alpha QA website!"
    return HttpResponse(info)
