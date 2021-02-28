"""
    实现内部排序算法
"""


class InnerSort(object):
    @classmethod
    def sift(cls, ls: list, low, high):
        i = low
        j = 2 * i
        tmp = ls[i]
        while j <= high:
            if j < high and ls[j] < ls[j + 1]:
                j += 1
            if tmp < ls[j]:
                ls[i] = ls[j]
                i = j
                j = 2 * i
            else:
                break
        ls[i] = tmp

    @classmethod
    def heap_sort(cls, ls: list, n):
        """ 堆排序 \n
        从 1 开始，数组大小为 n+1 \n
        :param ls:
        :param n: 传入为数组元素个数
        :return:
        """
        i = n // 2
        while i >= 1:
            cls.sift(ls, i, n)
            i -= 1
        i = n
        while i >= 2:
            ls[1], ls[i] = ls[i], ls[1]
            cls.sift(ls, 1, i-1)
            i -= 1


def main():
    ls = [100, 6, 8, 7, 9, 0, 1, 3, 2, 4, 5]
    InnerSort.heap_sort(ls, len(ls) - 1)
    print(ls)


if __name__ == '__main__':
    main()
