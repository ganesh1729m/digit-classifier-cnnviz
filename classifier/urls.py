from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_digit, name='predict'),
    path("save/", views.save_prediction, name="save_prediction"),
    path("report/", views.report_misclassification, name="report_misclassification"),
    # Login / Logout
    path('login/', auth_views.LoginView.as_view(template_name='classifier/login.html'), name='login'),
    # path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),


    # Register
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('rerun/<int:prediction_id>/', views.rerun_prediction, name='rerun_prediction'),
    path('rerun/<int:prediction_id>/', views.rerun_report, name='rerun_report'),


]