class Person(object):
    def __init__(self):
        self.__age = 19

    @property
    def age(self):
        print('---------get')
        return self.__age

    @age.setter
    def age(self, value):
        print("---------set")
        self.__age = value


p = Person()
print(p.age)
p.age = 10
print(p.age)


class Person1:
    # 当我们通过 “实例.属性=值”，给一个实例增加一个属性，或者修改一下属性值的时候
    # 都会调用这个方法，在这个方法内部，才会真正的把这个属性以及对应的数据存储到__dict__字典里面
    def __setattr__(self, key, value):
        print(key, value)
        # 1.判定key是否是我们要设置的只读属性的名称
        if key == "age" and key in self.__dict__.keys():
            print("这个属性是只读属性，不能设置数据")
        # 2.如果不是只读属性的名称，则将它添加到这个实例里面去
        else:
            self.__dict__[key] = value
            # 这句话会导致出现死循环
            # self.key=value


p1 = Person1()
p1.age = 18
print(p1.age)
p1.age = 999
print(p1.age)
print(p1.__dict__)
