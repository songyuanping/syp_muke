import sys
class Person:
    __personCount=0
    # 创建实例时执行的方法
    # def __new__(cls, *args, **kwargs):
    #     print('类进行创建')
    def __init__(self):
        print('计数+1')
        self.__class__.__personCount+=1
    def __del__(self):
        print('计数-1')
        self.__class__.__personCount-=1
    @classmethod
    def log(cls):
        print('实例个数：',cls.__personCount)
# p1=Person()
# print(p1)
# Person.log()
# p2=Person()
# Person.log()
# del p1
# Person.log()

p1=Person()
# 调用该函数会自动给引用计数加1
print(sys.getrefcount(p1))
p2=p1
print(sys.getrefcount(p2))
del  p1
print(sys.getrefcount(p2))
del p2
# print(sys.getrefcount(p2))

import gc
# (700,10,10)含义是：当创建的对象个数减去消亡的对象个数大于700时进行垃圾回收机制
# 且当0代回收执行10次后才执行一次1代和0代的共同回收，当1代回收进行了10次之后才进行1次2代、1代和0代的共同回收
print(gc.get_threshold())
gc.set_threshold(200,5,5)
print(gc.get_threshold())
# 手动触发垃圾回收机制，可以填入要执行的垃圾回收的代数（0、1、2），不填则执行一次完全意义上的垃圾回收
gc.collect()















