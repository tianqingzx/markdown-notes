"""
    这里主要实现各种图的存储结构，用于被继承和使用。
"""


class MGraph(object):
    """ 图的邻接矩阵存储 """
    def __init__(self, array, n):
        self.edges = array
        self.vexNum = n


class ArcNode(object):
    """ 图的邻接表存储 """
    def __init__(self, adj_vex):
        self.next = None
        self.adjVex = adj_vex


class AdjList(object):
    def __init__(self, data):
        self.data = data
        self.first = None


class ALGraph(object):
    def __init__(self, matrix):
        assert len(matrix) == len(matrix[0]), "输入不是一个矩阵！"
        self.vertices = []
        self.vexNum = len(matrix)
        self.arcNum = 0
        for i in range(self.vexNum):
            adj_list = AdjList(i + 1)
            p = None
            for j in range(self.vexNum):
                if matrix[i][j] != 0:
                    p = ArcNode(j + 1)
                    # 采用头插法插入结点
                    p.next = adj_list.first
                    adj_list.first = p
                    self.arcNum += 1
            self.vertices.append(adj_list)

    def display(self):
        p = None
        for index, adj_list in enumerate(self.vertices):
            p = adj_list.first
            print("{}: ".format(index + 1), end="")
            while p is not None:
                print("{}".format(p.adjVex), end=" ")
                p = p.next
            print("")


def main():
    ls = [
        [0, 1, 0],
        [2, 0, 0],
        [0, 3, 0]
    ]
    graph = ALGraph(ls)
    graph.display()


if __name__ == '__main__':
    main()
