from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('ML', views.image_classification_view, name='image-classifier'),
]
