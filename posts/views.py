from django.shortcuts import render, get_object_or_404, redirect, HttpResponse
from .models import Post
from .forms import PostForm
from django.http import JsonResponse
from itertools import groupby
import ContentToFont
from django.core.paginator import Paginator
import json
from django.views.decorators.csrf import csrf_exempt

def index(request):
    post_list = Post.objects.order_by('-pub_date')
    context = {'post_list': post_list}
    return render(request, 'posts/index.html', context)

def detail(request, post_id):
    post = get_object_or_404(Post, pk=post_id)
    return render(request, 'posts/detail.html', {'post': post})

def create(request):
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save()

            # 텍스트 기반 폰트 이름을 순서대로 할당하고 저장하기
            font_names_txt = ContentToFont.TexttoFont(post.content_text)[:3]  # case5 리스트의 0번부터 2번까지의 값 가져오기
            post.font_default = font_names_txt[0]
            post.font_name = font_names_txt[0]
            post.font_name2 = font_names_txt[1]
            post.font_name3 = font_names_txt[2]

            # 이미지 경로
            src = post.image.path
            font_names_img = ContentToFont.modelCompare("face", src)[:2]  # case5 리스트의 0번과 1번 값 가져오기

            # 이미지 기반 폰트 이름을 순서대로 할당하고 저장하기
            post.font_name_image = font_names_img[0]
            post.font_name_image2 = font_names_img[1]

            post.save()
            return redirect('index')

    else:
        form = PostForm()
    return render(request, 'posts/create.html', {'form': form})

def update(request, post_id):
    post = get_object_or_404(Post, pk=post_id)

    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            post = form.save()

            # 텍스트 기반 폰트 이름을 순서대로 할당하고 저장하기
            font_names_txt = ContentToFont.TexttoFont(post.content_text)[:3]  # case5 리스트의 0번부터 2번까지의 값 가져오기
            post.font_default = font_names_txt[0]
            post.font_name = font_names_txt[0]
            post.font_name2 = font_names_txt[1]
            post.font_name3 = font_names_txt[2]

            # 이미지 경로
            src = post.image.path
            font_names_img = ContentToFont.modelCompare("face", src)[:2]  # case5 리스트의 0번과 1번 값 가져오기

            # 이미지 기반 폰트 이름을 순서대로 할당하고 저장하기
            post.font_name_image = font_names_img[0]
            post.font_name_image2 = font_names_img[1]

            post.save()
            return redirect('detail', post_id)

    else:
        form = PostForm(instance=post)
    return render(request, 'posts/update.html', {'form': form})

def delete(request, post_id):
    post = Post.objects.get(pk=post_id)
    post.delete()
    return redirect('index')

@csrf_exempt
def save_font(request, post_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        font_name = data.get('font_name')
        post = get_object_or_404(Post, id=post_id)
        post.font_default = font_name  # font_default 값을 변경
        post.save()  # 변경된 값 저장
    return HttpResponse(status=200)

def get_fonts(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        post_ids = data.get('post_ids')
        font_data = {}
        for post_id in post_ids:
            post = get_object_or_404(Post, id=post_id)
            font_data[post_id] = post.font_name
        return HttpResponse(json.dumps(font_data), content_type='application/json')
