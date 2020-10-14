"""
    这里实现线索二叉树 \n
"""
from data_structure.stack_and_queue.sequence_stack import SqStack


class ThreadNode(object):
    def __init__(self, data):
        self.data = data
        self.l_child, self.r_child = None, None
        self.ltag, self.rtag = 0, 0


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

    @staticmethod
    def in_thread(p, pre):
        if p is not None:
            ThreadBiTree.in_thread(p.l_child, pre)
            if p.l_child is None:
                p.l_child = pre
                p.ltag = 1
            if pre is not None and pre.r_child is None:
                pre.r_child = p
                pre.rtag = 1
            pre = p
            ThreadBiTree.in_thread(p.r_child, pre)

    def create_in_thread(self):
        pre = None
        if self.head.l_child is not None:
            ThreadBiTree.in_thread(self.head.l_child, pre)
