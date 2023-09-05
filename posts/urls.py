from django.urls import path

from . import views

# app_name = 'posts'
urlpatterns = [
    # ex: /posts/
    path('', views.index, name='index'),
    # ex: /posts/create
    path('create/', views.create, name='create'),
    # ex: /posts/5/
    path('<int:post_id>/', views.detail, name='detail'),
    # ex: /posts/5/update
    path('<int:post_id>/update', views.update, name='update'),
    # ex: /posts/5/delete
    path('<int:post_id>/delete', views.delete, name='delete'),
    path('save_font/<int:post_id>/', views.save_font, name='save_font'),
    path('get_fonts/', views.get_fonts, name='get_fonts'),
]