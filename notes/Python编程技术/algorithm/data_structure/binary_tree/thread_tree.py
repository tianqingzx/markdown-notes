"""
    这里实现线索二叉树 \n
"""
from data_structure.stack_and_queue.sequence_stack import SqStack


class ThreadNode(object):
    def __init__(self, data):
        self.data = data
        self.l_child, self.r_child = None, None
        self.l_tag, self.r_tag = 0, 0


class ThreadBiTree(object):
    def __init__(self):
        self.head = ThreadNode(None)

    def create_binary_tree_by_str(self, chars: str):
        k, j = 0, 0
        p = None
        st = SqStack(10)
        while j < len(chars):
            ch = chars[j]
            if ch == "(":
                st.push(p)
                k = 1
            elif ch == ")":
                st.pop()
            elif ch == ",":
                k = 2
            else:
                p = ThreadNode(ch)
                if self.head.l_child is None:
                    self.head.l_child = p
                else:
                    node = st.pop()[1]
                    if k == 1:
                        node.l_child = p
                        st.push(node)
                    elif k == 2:
                        node.r_child = p
                        st.push(node)
            j += 1

    @classmethod
    def in_thread(cls, p, pre):
        """ 线索化函数
        递归算法实现（可考虑使用非递归算法）
        :return: pre
        """
        if p is not None:
            pre = cls.in_thread(p.l_child, pre)
            if p.l_child is None:  # 左子树为空，则建立前驱线索
                p.l_child = pre
                p.l_tag = 1
            if pre is not None and pre.r_child is None:  # 前驱结点后继线索
                pre.r_child = p
                pre.r_tag = 1
            pre = p
            pre = cls.in_thread(p.r_child, pre)
        return pre

    def create_in_thread(self):
        """ 主过程函数
        :return:
        """
        pre = self.head
        if self.head.l_child is not None:
            pre = self.in_thread(self.head.l_child, pre)
            pre.r_child = self.head
            pre.r_tag = 1
            self.head.r_child, self.head.r_tag = pre, 1

    @classmethod
    def first_node(cls, p):
        """ 寻找中序序列下第一个结点（最左下结点）
        :param p:
        :return:
        """
        while p.l_tag == 0:
            p = p.l_child
        return p

    @classmethod
    def next_node(cls, p):
        """ 中序序列下的后继
        :param p:
        :return:
        """
        if p.r_tag == 0:
            return cls.first_node(p.r_child)
        else:
            return p.r_child

    def in_order(self):
        """ 中序遍历
        :return:
        """
        p = self.first_node(self.head)
        while p.data is not None:
            print(p.data, end=" ")
            p = self.next_node(p)


def main():
    thread_tree = ThreadBiTree()
    thread_tree.create_binary_tree_by_str("A(B(C,D),F)")
    thread_tree.create_in_thread()
    thread_tree.in_order()


if __name__ == '__main__':
    main()
