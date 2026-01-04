from django.urls import path
from .views import FlowerPredictionAPI , MosaicAPI ,AnimeAPI

urlpatterns = [
    path('flower/', FlowerPredictionAPI.as_view(), name='flower-name'),
    path('mosaic/', MosaicAPI.as_view(), name='mosaic-art'),
    path('anime/', AnimeAPI.as_view(), name='anime-art'),
]
