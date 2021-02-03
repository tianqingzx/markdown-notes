"""
    这里实现图的遍历算法；深度优先遍历（DFS），广度优先遍历（BFS）\n
    所有的结点从 1 开始
"""
from data_structure.graph.graph_store_structure import ALGraph
from data_structure.stack_and_queue.sequence_queue import SqQueue


class ALGraphS(ALGraph):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.visited = []
        for i in range(self.vexNum + 1):
            self.visited.append(0)

    def dfs_(self, v):
        """ 递归算法 """
        self.visited[v] = 1
        print("{}".format(v), end=" ")
        p = self.adj_lists[v - 1].first
        while p is not None:
            if self.visited[p.adjVex] == 0:
                self.dfs_(p.adjVex)
            p = p.next

    def dfs(self):
        for i in range(self.vexNum + 1):
            self.visited[i] = 0
        for i in range(1, self.vexNum + 1):
            if self.visited[i] == 0:
                self.dfs_(i)

    def bfs_(self, v):
        """ 非递归算法 """
        qu = SqQueue(self.vexNum + 1)
        print("{}".format(v), end=" ")
        self.visited[v] = 1
        qu.en_queue(v)
        while qu.is_empty() is False:
            x = qu.de_queue()[1]
            p = self.adj_lists[x - 1].first
            while p is not None:
                if self.visited[p.adjVex] == 0:
                    print("{}".format(p.adjVex), end=" ")
                    self.visited[p.adjVex] = 1
                    qu.en_queue(p.adjVex)
                p = p.next
        print("")

    def bfs(self):
        for i in range(self.vexNum + 1):
            self.visited[i] = 0
        for i in range(1, self.vexNum + 1):
            if self.visited[i] == 0:
                self.bfs_(i)


def main():
    ls = [
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    graph = ALGraphS(ls)
    graph.dfs()


if __name__ == '__main__':
    main()
