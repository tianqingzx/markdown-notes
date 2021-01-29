"""
这是一个包
一些编写规范：\n
    1、一般不使用本实例的方法定义成静态方法@staticmethod，内部调用静态方法使用 类名.方法名\n
    2、需要使用实例本身属性和方法的，定义成一般方法self\n
    3、需要使用类属性和类方法的，定义成类方法@classmethod，主要是一般类名打起来太麻烦了\n
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
