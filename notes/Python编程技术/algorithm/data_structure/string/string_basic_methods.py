"""
    这里是关于串的定义，和基本的操作 \n\n

    StrAssign(chars)：赋值操作，将串赋值为 chars \n
    StrCompare(S, T)：比较操作。若S>T，则返回值>0；若S=T，则返回值=0；若S<T，则返回值<0 \n
    StrLength(S)：求串长。返回串S的元素个数 \n
    SubString(S, pos, len)：求子串。返回串S的第pos哥字符串起长度为len的子串 \n
    Concat(S1, S2)：串联接。用T返回由S1和S2联接而成的新串 \n
"""


class String(object):
    """
    匹配的模式串使用普通 str 类型 \n
    关于字符串的第一个字符：从1开始，全部考虑外部处理，同时输出也是从1开始 \n
    """
    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.ch: str = ""
        self.length: int = 0

    def str_assign(self, chars: str):
        if len(chars) <= self.max_size:
            self.ch = chars
            self.length = len(chars)
        else:
            self.ch = chars[:self.max_size]
            self.length = self.max_size

    def str_compare(self, str_: str) -> int:
        i = 0
        while i < self.length and i < len(str_):
            if self.ch[i] > str_[i]:
                return 1
            elif self.ch[i] < str_[i]:
                return -1
            i += 1
        if i >= self.length and i >= len(str_):
            return 0
        elif i >= self.length:
            return -1
        elif i >= len(str_):
            return 1

    def str_length(self) -> int:
        return len(self.ch)

    def sub_string(self, pos: int, length: int):
        pos -= 1
        sub = String(self.max_size)
        sub.str_assign(self.ch[pos:pos+length])
        return sub

    def index_1(self, model_str: str) -> int:
        """
        基于基本操作的暴力匹配实现算法 \n
        :param model_str:
        :return:
        """
        i = 1
        n = self.str_length()
        m = len(model_str)  # 需要再考虑一下是否将模式串也改为String类
        while i <= (n - m + 1):
            sub = self.sub_string(i, m)
            if sub.str_compare(model_str) != 0:
                i += 1
            else:
                return i
        return 0

    def direct_match(self, model_str: str) -> int:
        """
        基于暴力匹配算法实现 \n
        :param model_str:
        :return:
        """
        i = 0
        j = 0
        while i < self.length and j < len(model_str):
            if self.ch[i] == model_str[j]:
                i += 1
                j += 1
            else:
                i = i - j + 1
                j = 0
        if j >= len(model_str):
            return i - len(model_str) + 1
        else:
            return 0

    @staticmethod
    def get_kmp_next(model_str: str) -> list:
        """
        基于KMP算法，计算优化后的 nextval 数组 \n
        这里是静态方法 \n
        :param model_str:
        :return:
        """
        i, j = 0, -1
        next_list = list()
        next_list.append(-1)
        while i < len(model_str) - 1:
            if j == -1 or model_str[i] == model_str[j]:
                i += 1
                j += 1
                # next_list.append(j)  # 这里使用优化后的 next 数组
                if model_str[i] != model_str[j]:
                    next_list.append(j)
                else:
                    next_list.append(next_list[j])
            else:
                j = next_list[j]
        return next_list

    def kmp_match(self, model_str: str) -> int:
        """
        KMP算法，时间复杂度 O(m+n)
        :param model_str:
        :return:
        """
        i, j = 0, 0
        next_list = self.get_kmp_next(model_str)
        while i < self.length and j < len(model_str):
            if j == -1 or self.ch[i] == model_str[j]:
                i += 1
                j += 1
            else:
                j = next_list[j]
        if j >= len(model_str):
            return i - len(model_str) + 1
        else:
            return 0


def main():
    str_my = String(12)
    str_my.str_assign("hello world")
    pos = str_my.kmp_match("lo")
    print(str_my.ch)
    print(pos)


if __name__ == '__main__':
    main()
