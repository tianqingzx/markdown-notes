"""
    这里主要实现各种图的存储结构，用于被继承和使用。\n
    所有的结点从 1 开始
"""
from data_structure.stack_and_queue.sequence_stack import SqStack


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


class VNode(object):
    def __init__(self, data):
        # 顶点信息
        self.data = data
        # 数据域，存放顶点入度
        self.count = 0
        self.first = None


class ALGraph(object):
    def __init__(self, matrix):
        assert len(matrix) == len(matrix[0]), "输入不是一个矩阵！"
        self.adj_lists = []
        self.vexNum = len(matrix)
        self.arcNum = 0
        for i in range(self.vexNum):
            adj_list = VNode(i + 1)
            p = None
            for j in range(self.vexNum):
                if matrix[i][j] != 0:
                    p = ArcNode(j + 1)
                    # 采用头插法插入结点
                    p.next = adj_list.first
                    adj_list.first = p
                    self.arcNum += 1
            self.adj_lists.append(adj_list)

    def display(self):
        p = None
        for index, adj_list in enumerate(self.adj_lists):
            p = adj_list.first
            print("{}: ".format(index + 1), end="")
            while p is not None:
                print("{}".format(p.adjVex), end=" ")
                p = p.next
            print("")

    def top_sort(self):
        """ 实现图的拓扑排序 \n
        这里的adj_lists从 0 开始；其它都是从 1 开始
        """
        st = SqStack(self.vexNum + 1)
        for i in range(self.vexNum):
            self.adj_lists[i].count = 0
        for i in range(self.vexNum):
            p = self.adj_lists[i].first
            while p is not None:
                self.adj_lists[p.adjVex - 1].count += 1
                p = p.next

        for i in range(self.vexNum):
            if self.adj_lists[i].count == 0:
                st.push(i + 1)

        while st.is_empty() is False:
            i = st.pop()[1]
            print("{}".format(i), end=" ")
            p = self.adj_lists[i - 1].first
            while p is not None:
                j = p.adjVex
                self.adj_lists[j - 1].count -= 1
                if self.adj_lists[j - 1].count == 0:
                    st.push(j)
                p = p.next


def main():
    ls = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0]
    ]
    graph = ALGraph(ls)
    graph.display()
    print("")
    graph.top_sort()


if __name__ == '__main__':
    main()
