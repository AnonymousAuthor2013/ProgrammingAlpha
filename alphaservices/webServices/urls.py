from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index',views.index,name="homepage"),
    path('ask-post',views.getAnswer,name='ask-question')
]