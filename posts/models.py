from django.db import models

# Create your models here.
import datetime

from django.db import models
from django.utils import timezone


class Post(models.Model):
    title_text = models.CharField(max_length=100)  # 제목
    content_text = models.TextField()  # 내용
    pub_date = models.DateTimeField(auto_now_add=True)  # 자동으로 오늘 날짜로 설정됨
    font_default = models.CharField(max_length=100, default="Default Font") #텍스트 디폴트=font_name
    font_name = models.CharField(max_length=100, default="Default Font") #텍스트 폰트명1
    font_name2 = models.CharField(max_length=100, default="Default Font") #텍스트 폰트명2
    font_name3 = models.CharField(max_length=100, default="Default Font") #텍스트 폰트명3
    font_name_image = models.CharField(max_length=100, default="Default Font") # 사진 폰트명1
    font_name_image2 = models.CharField(max_length=100, default="Default Font") # 사진 폰트명2
    image = models.ImageField(upload_to='post_images/', null=True, blank=True)

    def __str__(self):
        return self.title_text

