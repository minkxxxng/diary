# Generated by Django 4.2.1 on 2023-07-16 05:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0003_mymodel_delete_event'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='context_image',
            field=models.ImageField(blank=True, null=True, upload_to='posts/'),
        ),
    ]
