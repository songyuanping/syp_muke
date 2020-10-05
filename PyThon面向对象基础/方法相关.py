class Age:
    def __get__(self, instance, value):
        print('set', self, instance, value)
        return instance.v

    def __set__(self, instance, value):
        print('set', self, instance, value)
        instance.v = value

    def __delete__(self, instance):
        print('del')
        del instance.v


class Person:
    # age是所有类共享的属性
    age = Age()


p = Person()
p.age = 10
print(p.age)
p1 = Person()
p1.age = 11
print(p1.age)
del p.age
# print(p.age)
print(p1.age)
