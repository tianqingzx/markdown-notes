"""
    实现平衡二叉树AVLBTree
    https://www.cnblogs.com/hello-shf/p/11352071.html
"""
from data_structure.stack_and_queue.sequence_queue import SqQueue


class AVLBiNode(object):
    def __init__(self, key):
        self.key = key
        self.height = 1
        self.left = None
        self.right = None


class AVLBTree(object):
    def __init__(self):
        """
        头结点为一个指针，而非一个空的结点
        size记录整棵树的结点数
        """
        self.head = None
        self.size = 0

    @staticmethod
    def get_height(node):
        if node is None:
            return 0
        return node.height

    @staticmethod
    def get_balance_factor(node):
        if node is None:
            return 0
        return AVLBTree.get_height(node.left) - AVLBTree.get_height(node.right)

    @staticmethod
    def right_rotate(node):
        node_left = node.left
        sub_node_right = node_left.right

        node_left.right = node
        node.left = sub_node_right

        node.height = 1 + max(AVLBTree.get_height(node.left), AVLBTree.get_height(node.right))
        node_left.height = 1 + max(AVLBTree.get_height(node_left.left), AVLBTree.get_height(node_left.right))
        return node_left

    @staticmethod
    def left_rotate(node):
        """ 一次左旋调整
        :param node:
        :return:
        """
        node_right = node.right
        sub_node_left = node_right.left

        node_right.left = node
        node.right = sub_node_left

        node.height = 1 + max(AVLBTree.get_height(node.left), AVLBTree.get_height(node.right))
        node_right.height = 1 + max(AVLBTree.get_height(node_right.left), AVLBTree.get_height(node_right.right))
        return node_right

    def tree_add(self, node, key):
        # 到了叶结点直接返回新的值
        if node is None:
            self.size += 1
            return AVLBiNode(key)
        # 递归判断大小，选择插入的走势
        if key < node.key:
            node.left = self.tree_add(node.left, key)
        elif key > node.key:
            node.right = self.tree_add(node.right, key)

        # 从下往上依次修改各个结点的高度
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

        # 计算当前结点平衡因子，同时进行调整
        balance_factor = self.get_balance_factor(node)
        if balance_factor > 1:
            if self.get_balance_factor(node.left) >= 0:
                return self.right_rotate(node)
            elif self.get_balance_factor(node.left) < 0:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        if balance_factor < -1:
            if self.get_balance_factor(node.right) <= 0:
                return self.left_rotate(node)
            elif self.get_balance_factor(node.right) > 0:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)

        return node

    def level_order(self):
        """
        层次遍历 \n
        :return:
        """
        queue = SqQueue(10)
        queue.en_queue(self.head)
        while queue.is_empty() is False:
            p = queue.de_queue()[1]
            print(str(p.key), end=" ")
            if p.left is not None:
                queue.en_queue(p.left)
            if p.right is not None:
                queue.en_queue(p.right)


def main():
    tree = AVLBTree()
    for key in [1, 2, 3, 4, 5]:
        tree.head = tree.tree_add(tree.head, key)
    tree.level_order()


if __name__ == '__main__':
    main()
