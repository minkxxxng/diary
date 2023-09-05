from __future__ import print_function
from google.cloud import vision
# from google.cloud.vision import types
from google.cloud.vision_v1 import types
import os
import io


def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "

    return result.strip()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'diary-396608-19e43b287fb2.json'
client = vision.ImageAnnotatorClient()


filenames = os.path.join(os.path.dirname(__file__), 'img.png')

with io.open(filenames, 'rb') as image_file:
    content = image_file.read()


image = types.Image()
image = vision.Image(content=content)
response = client.label_detection(image=image)

imagekeywordlist=[]

for label in response.label_annotations:
    imagekeywordlist.append(label.description)

imagekeyword = listToString(imagekeywordlist)

print(imagekeyword)







