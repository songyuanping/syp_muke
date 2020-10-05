class Person:
    def __init__(self):
        self.cache = {}

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __getitem__(self, item):
        return self.cache[item]

    def __delitem__(self, key):
        del self.cache[key]


p = Person()
p["name"] = 'jack'
print(p["name"])
print(p.__dict__)
del p['name']
print(p.__dict__)


class Person1:
    def __init__(self, age):
        self.age = age
        self.cache = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.cache[key] = value

    def __getitem__(self, item):
        return self.cache[item]

    def __delitem__(self, key):
        del self.cache[key]

    # 大于等于
    def __ge__(self, other):
        print('ge')
        return self.age == other.age

    # 不等于
    def __ne__(self, other):
        print(other)
        pass

    # 小于，采用调换参数的方式进行大于号的比较
    def __lt__(self, other):
        print('lt')
        print(self.age)
        print(other.age)
        return self.age < other.age
    # def __gt__(self, other):
    #     pass


p = Person1(18)
p[0:6:2] = ['a', 'b', 'c']
print(p.cache)
p2 = Person1(19)
print(p != p2)
print(p < p2)
print(p > p2)
import functools


# 生成所有的比较函数
@functools.total_ordering
class Person2:
    def __init__(self, age):
        self.age = age
        self.cache = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    # 大于等于
    def __eq__(self, other):
        print('ge')
        return self.age == other.age

    # 小于，采用调换参数的方式进行大于号的比较
    def __lt__(self, other):
        print('lt')
        print(self.age)
        print(other.age)
        return self.age < other.age


print('--------------------')
p1 = Person2(18)
p2 = Person2(20)
print(p1 >= p2)
print(Person2.__dict__)


class Person3:
    def __init__(self):
        self.result = 0

    def __getitem__(self, item):
        print('getitem')
        self.result += 1
        if self.result > 6:
            raise StopIteration('遍历结束！')
        return self.result

    # iter函数的优先级大于getitem函数的优先级。
    # 先调用iter在不断调用next函数实现迭代
    def __iter__(self):
        print('-----iter--------')
        self.result=1
        return self

    def __next__(self):
        self.result += 1
        if self.result > 6:
            raise StopIteration('遍历结束！')
        return self.result


p = Person3()
for i in p:
    print(i)
print('result:',p.result)
for i in p:
    print(i)
import collections
print(isinstance(p,collections.Iterator))