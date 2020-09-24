"""
    这是顺序队列的实现 \n
    这里也可以考虑实现双端队列，但是此处暂时不实现 \n
"""


class SqQueue(object):
    """
    1、初始状态（队空）：front = rear = 0 \n
    2、进队操作：队不满时，先送值到队尾元素，再将队尾指针加1 \n
    3、出队操作：队不空时，先取队头元素值，再将队头指针加1 \n
    ---------\n
    这里采用循环队列：\n
    队首指针进1：front = (front + 1) % max_size \n
    队尾指针进1：rear = (rear + 1) % max_size \n
    队列长度：(rear + max_size - front) % max_size \n
    ----------\n
    这里暂时采用牺牲一个空间来判满：\n
    队满：(rear + 1) % max_size == front \n
    队空：front == rear \n
    队列元素个数：(rear - front + max_size) % max_size \n
    """
    def __init__(self, max_size):
        self.data = list()
        self.front = self.rear = 0
        self.max_size = max_size

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
