import abc
class Animal(object,metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def bark(self):
        pass
class Dog(Animal):
    def bark(self):
        print('汪汪汪')
class Cat(Animal):
    def bark(self):
        print('喵喵猫')
def test(obj):
    obj.bark()
# a=Animal()
# test(a)
d=Dog()
test(d)
c=Cat()
test(c)