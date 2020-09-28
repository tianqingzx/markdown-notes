"""
    这里实现二叉树的链式存储结构
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


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
        # print('file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
        from data_structure.stack_and_queue.sequence_stack import SqStack
        k, j = 0, 0
        p = None
        st = SqStack(20)
        ch = chars[j]
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
                    flag, node = st.pop()
                    if k == 1:
                        node.l_child = p
                        st.push(node)
                    elif k == 2:
                        node.r_child = p
                        st.push(node)
            j += 1

    def pre_order(self, b):
        if b is not None:
            print(str(b.data))
            self.pre_order(b.l_child)
            self.pre_order(b.r_child)


def main():
    chars = "A(B,C)"
    bi_tree = BiTree()
    bi_tree.create_binary_tree_by_str(chars)
    bi_tree.pre_order(bi_tree.head.l_child)


if __name__ == '__main__':
    main()
