#### Django与Python兼容对照表

![django_1](F:\文档\Typora Files\markdown-notes\images\notes\python\django_1.png)



####  DateTimeField格式修改

```python
<td>{{ article.publish_date|date:"Y-m-d H:i:s" }}</td>
```



#### Django在PyCharm中objects显示不存在该关键字

![django_2](F:\文档\Typora Files\markdown-notes\images\notes\python\django_2.PNG)



#### redirect重定向数据传输问题

直接使用redirect没有办法传递参数，可以采用cookie和session两种方式来存储数据，然后可以在另一个试图读取

```python
def test(request):
    request.session['msg'] = "a message"
    return redirect('assetinfo:test')
```

相应的在HTML文件中写入

```python
...
{% if request.session.msg %}
<p>{{ request.session.msg }}</p>
{% endif %}
...
```



#### 403 CSRF跨域问题

在需要提交的form表单中加上如下代码

```python
<form ...>
{% csrf_token %}
...
</form>
```



#### os.path模块

[https://www.runoob.com/python/python-os-path.html](https://www.runoob.com/python/python-os-path.html)



