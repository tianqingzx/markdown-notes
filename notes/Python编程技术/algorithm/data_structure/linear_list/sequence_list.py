"""

"""


class SqList(object):
    def __init__(self, max_size):
        """
        初始化类实例
        :param max_size: 限制最大数组长度，模拟C语言数组
        """
        self.data = list()
        self.length = 0
        self.max_size = max_size

    def list_insert(self, i, e) -> bool:
        if i < 1 or i > (self.length + 1):
            return False
        if self.length >= self.max_size:
            return False
        for j in range(self.length, i-1, -1):
            self.data[j] = self.data[j-1]
        self.data[i-1] = e
        self.length += 1
        return True

    def list_delete(self, i):
        """

        :param i:
        :return: True | False, e | None
        """
        if i < 1 or i > self.length:
            return False, None
        e = self.data[i-1]
        for j in range(i, self.length):  # 将i之后的元素往前移动一位
            self.data[i-1] = self.data[i]
        self.length -= 1
        return True, e

    def locate_elem(self, e):
        """
        按元素值 e 进行查找，返回元素位序
        :param e:
        :return: (i + 1) | 0 元素位序
        """
        for i in range(self.length):
            if self.data[i] == e:
                return i + 1
        return 0

    def get_elem(self, i):
        """
        按元素位序 i 进行查找，返回元素值
        :param i:
        :return: True | False, e | None
        """
        if i < 1 or i > self.length:
            return False, None
        return True, self.data[i-1]

    def print_list(self):
        """
        输出顺序表中元素的值
        :return:
        """
        pass
