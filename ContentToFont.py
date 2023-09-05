#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

# In[1]:


#!pip install --upgrade tensorflow
#!pip install openpyxl
#!pip install deepface

# In[2]:


from deepface import DeepFace
import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
from sklearn.metrics.pairwise import cosine_similarity as cs

# In[3]:


from google.cloud import vision
# from google.cloud.vision import types
from google.cloud.vision_v1 import types
import os
import io

# In[4]:


#!pip install --upgrade ibm-watson

# In[5]:


import sqlite3

# In[6]:


def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "

    return result.strip()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'diary-396608-19e43b287fb2.json'
client = vision.ImageAnnotatorClient()

def GoogleCloudVision(filenames):

    with io.open(filenames, 'rb') as image_file:
        content = image_file.read()


    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    imagekeywordlist=[]

    for label in response.label_annotations:
        imagekeywordlist.append(label.description)

    imagekeyword = listToString(imagekeywordlist)

    return imagekeyword


# In[7]:


GoogleCloudVision("media/post_images/pic02.jpg")

# In[ ]:



import json

from ibm_watson import LanguageTranslatorV3, NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, EmotionOptions


authenticator = IAMAuthenticator('TBAPTVmwME9Po7-dcStbK-biFyqAeO8o23C2lN3DhqU4')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)

natural_language_understanding.set_service_url('https://api.jp-tok.natural-language-understanding.watson.cloud.ibm.com/instances/d97ac414-3d26-4849-82df-c21ebe6126db')


#한글 번역 기능 추가

# IBM Watson API 인증 정보 입력
language_translator_apikey = 'lUz7ZcCkdQLTEYyork7DXN2_R32G7dJMs5c3o2RROBxa'
language_translator_url = 'https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/424015bb-071a-4f07-9c71-330157a2971f'

natural_language_understanding_apikey = 'u0Vm7tToxAhCniyoKGj1U_DIxQ6g5fxc9--w8kjlI3jH'
natural_language_understanding_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/c3598d83-6802-45bd-8e96-f257269c018f'

# Watson Language Translator 인스턴스 생성
language_translator_authenticator = IAMAuthenticator(language_translator_apikey)
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    authenticator=language_translator_authenticator
)
language_translator.set_service_url(language_translator_url)

# Watson Natural Language Understanding 인스턴스 생성
nlu_authenticator = IAMAuthenticator(natural_language_understanding_apikey)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=nlu_authenticator
)
natural_language_understanding.set_service_url(natural_language_understanding_url)

# 한국어 텍스트를 감정 분석하는 함수 정의
def analyze_emotion_korean(text):
    # 한국어를 영어로 번역
    translation = language_translator.translate(
        text=text,
        model_id='ko-en'
    ).get_result()
    translated_text = translation['translations'][0]['translation']

    # 번역된 영어 텍스트를 감정 분석
    response = natural_language_understanding.analyze(
        text=translated_text,
        features=Features(emotion=EmotionOptions())).get_result()
    emotion = response['emotion']['document']['emotion']
    return emotion



def IBMWatsonText(user_input_text):
# 한국어 텍스트를 감정 분석하여 결과 저장
      emotion_result = analyze_emotion_korean(user_input_text)
      return emotion_result

def IBMWatsonImage(user_input_image):
      response = natural_language_understanding.analyze(
      html=user_input_image,
      features=Features(emotion=EmotionOptions())).get_result()
      emotion = response['emotion']['document']['emotion']
      return emotion






# In[ ]:


def fkdf_prepro(df):
    df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    return df
#데이터 읽어오기
FontVAD="FontKeywordVAD_update.xlsx"

# 폰트 키워드 값
fkAll_df=pd.read_excel(FontVAD, sheet_name="Sheet1", engine='openpyxl')
font_list=fkAll_df['Unnamed: 0'].values.tolist()
fkAll_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# 폰트 별 키워드에 대한 V 값
fkV_df=pd.read_excel(FontVAD, sheet_name="V", engine='openpyxl')
fkV_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
keyword_list=fkV_df.columns.tolist()

# 폰트 별 키워드에 대한 A 값
fkA_df=pd.read_excel(FontVAD, sheet_name="A", engine='openpyxl')
fkA_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# 폰트 별 키워드에 대한 D 값
fkD_df=pd.read_excel(FontVAD, sheet_name="D", engine='openpyxl')
fkD_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# 폰트 키워드에서 VAD 값으로 변환한 값 호출
fVAD_df=pd.read_excel(FontVAD, sheet_name="FontVAD", engine='openpyxl')
fVAD_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# 각 감정 분류 기준에 따르 피어슨 상관계수 값 호출
kCorr_df=pd.read_excel("keyword_corr.xlsx", sheet_name="Sheet2", engine='openpyxl')
kCorr_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# In[ ]:


#데이터 프레임 딕셔너리로 변경
font_keyword_vad={}
for i in range(len(font_list)):
    tmp2={}
    for k in keyword_list:
        tmp=[]
        tmp.append(fkV_df.iloc[i][k])
        tmp.append(fkV_df.iloc[i][k])
        tmp.append(fkA_df.iloc[i][k])
        tmp.append(fkD_df.iloc[i][k])
        tmp2[k]=tmp

    font_keyword_vad[font_list[i]]=tmp2

#

# In[ ]:


class FontByEmotion:
  def __init__(self, emotion):
    self.emotion=emotion
     # 콘텐츠 감정 분류 기준의 VAD 값
    self.basic_emotion_vad={"angry":[0.122, 0.830, 0.604],"disgust":[0.052, 0.775, 0.317], "fear":[0.073, 0.840, 0.293],
                   "happy":[1.000, 0.735, 0.772], "sad":[0.225, 0.333, 0.149], "surprise":[0.875, 0.875, 0.562],
                   "neutral":[0.469, 0.184, 0.357],"anger":[0.167, 0.865, 0.657], "contempt":[0.206, 0.636, 0.396], "happiness":[0.96, 0.732, 0.85],
                   "sadness":[0.052, 0.288, 0.164], "joy":[0.98, 0.824, 0.794], "trust":[0.888, 0.547, 0.741] }
    self.emotion_list=list(emotion.keys())
    self.overallVAD=self.EmotionVAD_overall(emotion)
    self.emotionVAD=self.EmotionVAD(emotion)

  def euclidean_cal(self, list1, list2): #단순 유클리디안 거리 계산
    sum=0
    for i in range(len(list1)):
      sum+=(list2[i]-list1[i])**2

    return math.sqrt(sum)

  def sorted_dict(self,dict):
    sorted_list = sorted(dict.items(), key=lambda x:x[1])
    return sorted_list

  #콘텐츠 감정을 VAD 값으로 변경(하나의 대표 값)
  def EmotionVAD_overall(self, emotion):
    overall_vad=[]
    for i in range(0,3):  #V, A, D
        sum_product=0
        for e in self.emotion_list: # 각 감정의 VAD 값과 추출된 감정의 수치 합침
            sum_product+=self.basic_emotion_vad[e][i]*emotion[e]

        vad_value=1/(1+math.exp(-0.5*(sum_product-4.75))) #딥러닝 계산을 레퍼런스하여 만든 계산식
        overall_vad.append(vad_value)

    return overall_vad

  # 콘텐츠 감정 별 VAD 값
  def EmotionVAD(self, emotion):
    emotion_vad={}
    for e in self.emotion_list:
      vad_list=[]
      for i in range(0,3):
        tmp=self.basic_emotion_vad[e][i]*emotion[e]
        vad_list.append(tmp)
      emotion_vad[e]=vad_list

    return emotion_vad
  def corrsim(self):
    dataset=[]

    #폰트키워드 별로 감정과 글꼴의 상관계수를 더하여 계산
    for i in range(len(fkAll_df.columns)):
        sum=0
        for e in self.emotion_list:
            sum+=self.emotion[e]*kCorr_df.iloc[i][e]

        if sum<0:
            sum=0 #피어슨 상관계수가 음의 방향이면 0으로 처리(변경해도 됨)

        dataset.append(sum)

    #코사인 유사도 계산을 위해 형식 일치 시킨 후 계산
    EmotionCorr_df=pd.DataFrame([dataset], columns=keyword_list, dtype=float)
    cos_sim=pd.DataFrame(cs(fkAll_df, EmotionCorr_df))
    #sim_value_list=cs.to_dict()

    #폰트 랭킹 소팅
    sorted_cossim=cos_sim[0].argsort()

    font_ranking=[]
    for i in range(0,len(sorted_cossim)):
        font_ranking.append(font_list[sorted_cossim[i]])

    font_ranking.reverse()
    return font_ranking

# In[ ]:


#얼굴인식 후 폰트 추천해주는 코드
def face_emotion(path):
    analysis = DeepFace.analyze(img_path = path, actions = ["emotion"])
    face_emotions=analysis[0]["emotion"]
    emotion = {}
    for e in face_emotions:
        emotion[e]=face_emotions[e]*0.01

    print(emotion)
    return emotion


def modelCompare(flag, src):
   if flag=="face":
       try:
         emotion=face_emotion(src)
         RecommModel=FontByEmotion(emotion)
         case5=RecommModel.corrsim()
         print(case5[0:5])
         # return case5[0]
         return case5

       except:
           fromGCV = GoogleCloudVision(src)
           test = IBMWatsonImage(fromGCV)
           RecommModel=FontByEmotion(test)
           case5=RecommModel.corrsim()
           print(case5[0:5])
           # return case5[0]
           return case5

def TexttoFont(user_text):
    user_text1 = IBMWatsonText(user_text)
    RecommModel=FontByEmotion(user_text1)
    case5=RecommModel.corrsim()
    print(case5[0:5])
    #return case5[0]
    return case5







con = sqlite3.connect("db.sqlite3")
cur = con.cursor()
result = cur.execute("select image from posts_post;")
rows = result.fetchall()
image_path_list=[]
for row in rows:
    image_path = row[0]
    image_path_list.append(image_path)

img = image_path_list[0]

