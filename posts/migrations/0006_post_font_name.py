# Generated by Django 4.2.1 on 2023-07-16 06:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0005_remove_post_context_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='font_name',
            field=models.CharField(default='Default Font', max_length=100),
        ),
    ]