"""
    包含栈相关的自我实现的库 \n
    包含队列相关的库 \n\n

    顺序队相关：\n
    1、初始状态（队空）：front = rear = 0 \n
    2、进队操作：队不满时，先送值到队尾元素，再将队尾指针加1 \n
    3、出队操作：队不空时，先取队头元素值，再将队头指针加1 \n
    ---------\n
    这里采用循环队列：\n
    队首指针进1：front = (front + 1) % max_size \n
    队尾指针进1：rear = (rear + 1) % max_size \n
    队列长度：(rear + max_size - front) % max_size \n
    ----------\n
    这里暂时采用牺牲一个空间来判满：\n
    队满：(rear + 1) % max_size == front \n
    队空：front == rear \n
    队列元素个数：(rear - front + max_size) % max_size \n
"""