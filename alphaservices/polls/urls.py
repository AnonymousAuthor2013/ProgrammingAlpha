from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask-post',views.getAnswer,name='ask-question')
]
