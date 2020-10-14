"""
    这里实现二叉树的链式存储结构
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
# print('file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
from data_structure.stack_and_queue.sequence_stack import SqStack
from data_structure.stack_and_queue.sequence_queue import SqQueue


class BiTNode(object):
    def __init__(self, data):
        self.data = data
        self.l_child = None
        self.r_child = None


class BiTree(object):
    """
    这里创造一个带头节点的二叉树，头节点的左孩子指向根结点 \n
    """
    def __init__(self):
        self.head = BiTNode(None)

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
                p = BiTNode(ch)
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

    def post_order_(self, b):
        if b is not None:
            self.post_order_(b.l_child)
            self.post_order_(b.r_child)
            print(str(b.data), end=" ")

    def post_order(self):
        """
        统一接口，实现封装，具体实现不对外公布 \n
        :return:
        """
        self.post_order_(self.head.l_child)

    def post_order_2(self):
        """
        后序非递归算法实现 \n
        :return:
        """
        st = SqStack(10)
        p = self.head.l_child
        while True:
            while p is not None:
                st.push(p)
                p = p.l_child
            r = None
            flag = True
            while st.is_empty() is False and flag:
                p = st.get_top()[1]
                if p.r_child == r:
                    print(str(p.data), end=" ")
                    p = st.pop()[1]
                    r = p
                else:
                    p = p.r_child
                    flag = False
            if st.is_empty() is True:
                break
        print("")

    def level_order(self):
        """
        层次遍历 \n
        :return:
        """
        queue = SqQueue(10)
        queue.en_queue(self.head.l_child)
        while queue.is_empty() is False:
            p = queue.de_queue()[1]
            print(str(p.data), end=" ")
            if p.l_child is not None:
                queue.en_queue(p.l_child)
            if p.r_child is not None:
                queue.en_queue(p.r_child)


def main():
    chars = "A(B,C(D,E))"
    bi_tree = BiTree()
    bi_tree.create_binary_tree_by_str(chars)
    bi_tree.level_order()


if __name__ == '__main__':
    main()
