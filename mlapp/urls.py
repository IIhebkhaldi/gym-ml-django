from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('churn/', views.predict_churn, name='predict_churn'),
    path('clustering/', views.predict_clustering, name='predict_clustering'),
    path('satisfaction/', views.predict_satisfaction, name='predict_satisfaction'),
    path('health/', views.health_check, name='health_check'),
    path('debug/', views.debug_info, name='debug_info'),
    path('test/', views.test_logic, name='test_logic'),  # Ajoutez cette ligne
]

