class Person:
    def __init__(self):
        self.age = 1

    def __iter__(self):
        self.age = 1
        return self

    def __next__(self):
        self.age += 1
        if self.age > 6:
            raise StopIteration("stop")
        return self.age

    # 实现调用：pt = iter(p, 4)
    def __call__(self, *args, **kwargs):
        self.age += 1
        if self.age > 6:
            raise StopIteration("stop")
        return self.age


p = Person()
# 自动比较与4的大小关系
pt = iter(p, 4)
# 第一个参数是可迭代对象
pt = iter(p.__next__, 4)
# print(isinstance(p,pt))
for i in pt:
    print(i)
