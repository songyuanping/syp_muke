class Person(object):
    def __init__(self):
        self.__age = 19

    # 主要作用是，可以使用属性的方式来使用这个方法
    @property
    def age(self):
        return self.__age


p1 = Person()
print(p1.age)
# p1.age=233
class Person1(object):
    def __init__(self):
        self.__age=17
    def get_age(self):
        return self.__age
    def set_age(self,value):
        self.__age=value
    age=property(get_age,set_age)

p=Person1()
print(p.age)
p.age=99
print(p.age)
print(p.__dict__)