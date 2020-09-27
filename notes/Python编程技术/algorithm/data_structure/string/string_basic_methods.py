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
    关于字符串的第一个字符：从1开始，全部考虑外部处理 \n
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


def main():
    str_my = String(12)
    str_my.str_assign("hello world")
    pos = str_my.index_1("lo")
    print(str_my.ch)
    print(pos)


if __name__ == '__main__':
    main()
