"""
    这里用来实现栈的链式存储结构

    [pass]
"""


class LStNode(object):
    def __init__(self, data):
        self.data = data
        self.next = None


class LiStack(object):
    """
    1、不会出现栈满上溢的情况 \n
    2、采用单链表实现 \n
    3、规定链栈没有头节点 \n
    """
    def __init__(self):
        pass


def main():
    node = LStNode(1)
    node.next = LStNode(2)
    head = node
    while head is not None:
        print(head.data)
        head = head.next


if __name__ == '__main__':
    main()
