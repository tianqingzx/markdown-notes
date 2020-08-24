"""
"""


# 此处为单链表的结点
class SLNode(object):
    def __init__(self, data):
        self.data = data
        self.next = None


# 此处为双链表的结点
class DLNode(object):
    def __init__(self, data):
        self.data = data
        self.prior = None
        self.next = None


class SingleLinkList(object):
    def __init__(self):
        self.head = SLNode(None)

    def list_insert(self, i, e):
        """
        algorithm: 在单链表中找到第 i-1 个结点，然后由 p 指向它。
                若存在这样的结点，将值为 e 的结点插入到 p 所指结点之后。
        :param i:
        :param e:
        :return:
        """
        j = 0
        p = self.head
        if i <= 0:
            return False
        while j < i-1 and p is not None:
            j += 1
            p = p.next
        if p is None:
            return False
        else:
            s = SLNode(e)
            s.next = p.next
            p.next = s
            return True

    def list_delete(self, i):
        pass

    def get_elem(self, i):
        pass

    def locate_elem(self, e):
        pass

    def disp_list(self):
        p = self.head.next
        while p is not None:
            print(" {0} ".format(p.data), end="")
            p = p.next
        print("\n")


class DoubleLinkList(object):
    def __init__(self):
        self.head = DLNode(None)

    def list_insert(self, i, e):
        j = 0
        p = self.head
        if i <= 0:
            return False
        while j < i-1 and p is not None:
            j += 1
            p = p.next
        if p is None:
            return False
        else:
            s = DLNode(e)
            s.next = p.next
            if p.next is not None:
                p.next.prior = s
            s.prior = p
            p.next = s
            return True

    def list_delete(self, i):
        j = 0
        p = self.head
        if i <= 0:
            return False, None
        while j < i-1 and p is not None:
            j += 1
            p = p.next
        if p is None:
            return False, None
        else:
            q = p.next
            if q is None:
                return False, None
            e = q.data
            p.next = q.next
            if p.next is not None:
                p.next.prior = p
            del q
            return True, e


def main():
    link_list = SingleLinkList()
    for i in range(5):
        link_list.list_insert(1, i)
    link_list.disp_list()


if __name__ == '__main__':
    main()
