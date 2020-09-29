"""
    这是顺序队列的实现 \n
    这里也可以考虑实现双端队列，但是此处暂时不实现 \n
"""


class SqQueue(object):
    """
    这里采用循环队列：\n
    这里暂时采用牺牲一个空间来判满：\n
    """
    def __init__(self, max_size):
        self.data = list()
        self.front = self.rear = 0
        self.max_size = max_size
        for i in range(max_size):
            self.data.append(None)

    def is_empty(self) -> bool:
        if self.rear == self.front:
            return True
        else:
            return False

    def en_queue(self, e) -> bool:
        if (self.rear + 1) % self.max_size == self.front:
            return False
        self.data[self.rear] = e
        self.rear = (self.rear + 1) % self.max_size
        return True

    def de_queue(self):
        if self.rear == self.front:
            return False, None
        e = self.data[self.front]
        self.front = (self.front + 1) % self.max_size
        return True, e
