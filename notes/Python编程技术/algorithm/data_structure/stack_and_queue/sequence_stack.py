"""
    栈的基本操作 \n

    [共享栈未实现]
"""


class SqStack(object):
    def __init__(self, max_size):
        """
        初始化栈实例 \n
        :param max_size： 限制最大栈深
        """
        self.data = list()
        self.max_size = max_size
        self.top = -1

    def stack_empty(self) -> bool:
        if self.top == -1:
            return True
        else:
            return False

    def push(self, e) -> bool:
        """
        入栈操作 \n
        :param e: 入栈的元素
        :return:
        """
        if self.top == self.max_size - 1:
            return False
        self.top += 1
        self.data[self.top] = e
        return True

    def pop(self):
        if self.top == -1:
            return False, None
        e = self.data[self.top]
        self.top -= 1
        return True, e

    def get_top(self):
        if self.top == -1:
            return False, None
        e = self.data[self.top]
        return True, e


class SharedStack(object):
    """
    这是一个共享栈 \n
    两个顺序栈共享一个数组空间，栈底分别在两端 \n
    top0 = -1 时0号栈为空，top1 = max_size 时1号栈为空 \n
    top1 - top0 = 1 时栈满 \n

    [pass]
    """
    def __init__(self, max_size):
        self.top_0 = -1
        self.top_1 = max_size

