import math
import tensorflow as tf


def test_grad():
    x = tf.Variable(1.0)
    with tf.GradientTape() as t1:
        with tf.GradientTape() as t2:
            y = x * x * x
        dy_dx = t2.gradient(y, x)
        print(dy_dx)
    d2y_dx2 = t1.gradient(dy_dx, x)
    print(d2y_dx2)


def information_entropy(labels: list) -> float:
    """
    :param labels: 数组
    :return: ent
    """
    assert 0 != len(labels), 'labels 非空'
    counter = {}
    for y in labels:
        if str(y) not in counter.keys():
            counter[str(y)] = 1
        else:
            counter[str(y)] += 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / len(labels)
        ent -= p * math.log2(p)
    return ent


def main():
    ls = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(information_entropy(ls), len(ls))


if __name__ == "__main__":
    main()
