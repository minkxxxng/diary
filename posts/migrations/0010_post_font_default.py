# Generated by Django 4.2.1 on 2023-08-12 15:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0009_post_font_name2_post_font_name3_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='font_default',
            field=models.CharField(default='Default Font', max_length=100),
        ),
    ]