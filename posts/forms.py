from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        # fields = ['title_text', 'content_text','font_default', 'font_name', 'font_name_image', 'font_name2', 'font_name_image2', 'font_name3', 'image']
        fields = ['title_text', 'content_text', 'image']