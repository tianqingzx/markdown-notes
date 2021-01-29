"""
该库用来实现与二叉树相关的算法和数据结构 \n
-------\n
+ 只知道二叉树的先序和后序序列，无法唯一确定一棵二叉树 \n
+ 在含有n个结点的二叉树中，有n+1个空指针 \n
-------\n
树的一般结点和遍历方法可以抽象出来分别做为抽象结点和抽象树\n
+ TreeNode { var: {data|key:value, left, right}, method: {__init__} }\n
+ Tree { var: {head}, method: {__init__, level_order, pre_order, in_order} }\n
"""
