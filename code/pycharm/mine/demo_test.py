"""
    整体性测试
"""


def func_test(*args):
    i, s, f = args
    print(i, s, f)


if __name__ == '__main__':
    func_test(1, 's', 2.2)
