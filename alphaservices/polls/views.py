from django.shortcuts import render
from django.http import HttpResponse,HttpRequest
#from .qamodel import process
def process(q):return "No"
# Create your views here.

def index(request:HttpRequest):
    msg="hello, welcome to programming alpoha for AI engineers and learners !!!"
    return HttpResponse(msg)

def getAnswer(request:HttpRequest):
    request.encoding='utf-8'
    ans=None
    if 'q' in request.POST:
        question=request.POST['q']
        message = 'the quetsion you asked is ' + question
        ans=process(question)
    else:
        message = 'cannot answer blank questions!!!'

    reply ={}
    if request.POST:
        reply['answer'] = ans if ans else message

    print("request body=>",request.body)

    return render(request, "ask-post.html", reply)
