import json
import requests

# PyTorch用的包
import torch
import torch.nn as nn
import torch.optim as optim

# 自然语言处理相关的包
import re
import jieba  # 结巴分词器
from collections import Counter  # 搜集器，可以让统计词频更简单

# 绘图、计算用的包
import matplotlib.pyplot as plt
import numpy as np

# header = {"user-agent": "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) "
#                         "Chrome/20.0.1132.57 Safari/536.11"}

header = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/88.0.4324.182 Safari/537.36 Edg/88.0.705.81"}


# 在指定的url处获得评论
def get_comments(url_):
    comments = []
    # 打开指定页面
    resp = requests.get(url_, headers=header)
    resp.encoding = 'gbk'

    # 如果返回不是200则失败
    if resp.status_code != 200:
        return []

    # 获得内容
    content = resp.text
    if content:
        # 获得括号中的内容
        ind = content.find('(')
        s1 = content[ind + 1:-2]
        try:
            # 尝试利用json接口读取内容，并作json解析
            js = json.loads(s1)
            # 提取comments字段的内容
            comment_infos = js['comments']
        except:
            print('error')
            return []

        # 对每一条评论进行内容部分的抽取
        for comment_info in comment_infos:
            comment_content = comment_info['content']
            str1 = comment_content + '\n'
            comments.append(str1)
    return comments


good_comments = []

good_comment_url_templates = [
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=11993134&score=1'
    '&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12443890&score'
    '=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12998056&score'
    '=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12176278&score'
    '=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1 ',

    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=11993134&score=2'
    '&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12443890&score'
    '=2&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12998056&score'
    '=2&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/skuProductPageComments.action?callback=fetchJSON_comment98&productId=12176278&score'
    '=2&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1 '
]

j = 0
for good_comment_url_template in good_comment_url_templates:
    for i in range(20):
        url = good_comment_url_template.format(i)
        good_comments += get_comments(url)
        if i % 5 == 0:
            print('第{}条记录，总文本长度{}'.format(j, len(good_comments)))
        j += 1

# 将结果存储到good.txt文件中
with open('E:/ai_learning_resource/mood_classifier/bad.txt', 'w', encoding='utf-8') as fw:
    fw.writelines(good_comments)
