import abc


class Animal(object, metaclass=abc.ABCMeta):
    def __init__(self, name, age=1):
        self.name = name
        self.age = age

    @abc.abstractmethod
    def eat(self):
        pass

    @abc.abstractmethod
    def play(self):
        pass

    @abc.abstractmethod
    def sleep(self):
        pass


class Dog(Animal):
    def __init__(self, name, age):
        super(Dog, self).__init__(name, age)

    def eat(self):
        print('%s在吃饭' % self)

    def play(self):
        print('%s在玩' % self)

    def sleep(self):
        print('%s在睡觉' % self)

    def work(self):
        print('%s在看家' % self)

    def __str__(self):
        return '名字是{},年龄是{}的小狗'.format(self.name, self.age)


class Cat(Animal):
    def __init__(self, name, age):
        super(Cat, self).__init__(name, age)

    def eat(self):
        print('%s在吃饭' % self)

    def play(self):
        print('%s在玩' % self)

    def sleep(self):
        print('%s在睡觉' % self)

    def work(self):
        print('%s在捉老鼠' % self)

    def __str__(self):
        return '名字是{},年龄是{}的小猫'.format(self.name, self.age)


class Person(Animal):
    def __init__(self, name, pets, age=1):
        super(Person, self).__init__(name, age)

        self.pets = pets

    def eat(self):
        print('%s在吃饭' % self)

    def play(self):
        print('%s在玩' % self)

    def sleep(self):
        print('%s在睡觉' % self)
    def make_pets_work(self):
        for pet in self.pets:
            pet.work()
    def feed_pets(self):
        result=''
        for pet in self.pets:
            result+='[ '+pet.name+' , '+str(pet.age)+'岁 ]'
        print('{}在饲养{}'.format(self,result))
    def __str__(self):
        return '名字是{},年龄是{}的人'.format(self.name, self.age)
c=Cat('小喵',3)
# c.work()
d=Dog('小汪',6)
# d.work()
p=Person('小张',[c,d],10)
p.make_pets_work()
p.feed_pets()