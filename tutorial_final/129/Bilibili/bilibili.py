import requests
import json
import pandas as pd

class Video:
    def __init__(self, **kwargs):
        self.play = kwargs.get('play', 0)
        self.description = kwargs.get('description', None)
        self.pubdate = kwargs.get('pubdate', None)
        self.title = kwargs.get('title', None)
        self.review = kwargs.get('review', 0)
        # User id
        self.mid = kwargs.get('mid', None)
        self.tag = kwargs.get('tag', None)
        # Bullet count
        self.video_review = kwargs.get('video_review', None)
        self.author = kwargs.get('author', None)
        self.favorites = kwargs.get('favorites', None)
        self.duration = kwargs.get('duration', None)
        self.id = kwargs.get('id', None)



keywords = { 'view_type': 'hot_rank',
'order': 'click',
'copy_right' :'-1',
'cat_id': '21',
'page': 1,
'pagesize': 20,
'jsonp': 'jsonp',
'time_from': '20180305',
'time_to': '20180312',
'keyword': 'vlog'}

videos = []
page = 1
url = 'https://s.search.bilibili.com/cate/search?main_ver=v3&search_type=video&view_type=hot_rank&order=click&copy_right=-1&cate_id=21&pagesize=20&jsonp=jsonp&time_from=20180305&time_to=20180312&keyword=vlog&page=' # &_=1520796787867'

r = requests.get(url+str(page))
response = json.loads(r.text.decode("utf-8"))
num_pages = int(response['numPages'])

for i in range(1, num_pages+1):
    print i
    r = requests.get(url+str(i))
    response = json.loads(r.text.decode("utf-8"))
    for video in response['result']:
        videos.append(Video(**video))

print videos