"""
    这里实现链队
"""


class LQuNode(object):
    """"""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkQueue(object):
    """
    这是一个带有头节点的链队 \n
    [可以考虑采用其它的实现方式]
    """
    def __init__(self):
        self.front = self.rear = LQuNode(None)

    def is_empty(self) -> bool:
        if self.front == self.rear:
            return True
        else:
            return False

    def en_queue(self, e):
        s = LQuNode(e)
        self.rear.next = s
        self.rear = s

    def de_queue(self):
        if self.front == self.rear:
            return False, None
        p = self.front.next
        e = p.data
        self.front.next = p.next
        if self.rear == p:
            self.rear = self.front
        return True, e


def main():
    lq = LinkQueue()
    lq.en_queue(1)
    lq.en_queue(2)
    lq.en_queue(3)
    while lq.is_empty() is False:
        print(lq.de_queue())


if __name__ == '__main__':
    main()
