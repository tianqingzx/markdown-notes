"""
    这里实现并查集
"""


class UFSets(object):
    def __init__(self, size):
        self.size = size
        self.sets = []
        for i in range(size):
            self.sets.append(-1)

    def find(self, index):
        while self.sets[index] >= 0:
            index = self.sets[index]
        return index

    def union(self, root1, root2):
        """ 将root2连接到root1下
        :param root1:
        :param root2:
        :return:
        """
        assert root1 == root2, "root1 和 root2 表示不同的子集合，不能相同！"
        self.sets[root2] = root1

    def display(self):
        print("Sets[")
        for index, el in enumerate(self.sets):
            print("{}: {}".format(index, el), end=" ")
        print("]")
