# Generated by Django 4.2.1 on 2023-08-12 15:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0008_post_font_name_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='font_name2',
            field=models.CharField(default='Default Font', max_length=100),
        ),
        migrations.AddField(
            model_name='post',
            name='font_name3',
            field=models.CharField(default='Default Font', max_length=100),
        ),
        migrations.AddField(
            model_name='post',
            name='font_name_image2',
            field=models.CharField(default='Default Font', max_length=100),
        ),
    ]