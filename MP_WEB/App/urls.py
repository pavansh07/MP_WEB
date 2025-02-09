from django.urls import path
from . import views

urlpatterns = [
    # Home page
    path('', views.home_view, name='home'),

    # Graphical Method URLs
    path('graphical_method/', views.graphical_method_view, name='graphical_method'),
    path('graphical_method/solve/', views.graphical_solve_view, name='graphical_solve'),
    path('graphical_method/steps/', views.graphical_steps_view, name='graphical_steps'),
    path('graphical_method/application/', views.graphical_application_view, name='graphical_application'),
    
    # Simplex Method URLs
    path('simplex/', views.simplex_method_view, name='simplex_method'),
    path('simplex/steps/', views.simplex_steps_view, name='simplex_steps'),
    path('simplex/solve/', views.simplex_solve_view, name='simplex_solve'),
    path('simplex/application/', views.simplex_application_view, name='simplex_application'),
]