class Person:
    """
    关于这个类的描述，类的作用，类的构造函数；类属性的描述
    Attributes:
        num: int 代表人的个数
    """
    num=0
    # 此处可以访问类属性和实例属性
    def eat(self, food):
        """
        这个方法的作用效果
        :param food: 参数的含义，参数的类型int，是否有默认值
        :return: 返回结果的含义，返回数据的类型int
        """
        print('在吃饭，', self, food)

    # p.eat2()进行调用时，解释器会自动传送对象本身作为参数，此时会出错
    # def eat2():
    #     print('eat2')


    def entity(self):
        print('这是一个实例方法！')
        print(self)
    # 其子类在调用时，传入的cls为子类
    # 其作用主要是访问类属性，并对其进行操作
    @classmethod
    def class_method(cls,num):
        print('这是一个类方法！')
        print(cls,num)

    @staticmethod
    def static():
        # 可以访问到类的属性，但静态方法一般对应于不访问类属性和实例属性的场景
        print(Person.num)
        print('这是一个静态方法！')
class A(Person):
    pass

p = Person()
p.age = 12
p.entity()
print(p)
# p.class_method()
p.static()
# Person.class_method()
# Person.static()
# 把实例方法当做函数调用
print(Person.eat)
Person.eat(123, '土豆')
A.class_method(100)
def run(self):
    print(self)

xxx=type('Dog',(),{'count':0,'run':run})
print(xxx)
print(xxx.__dict__)
d=xxx()
print(d)
d.run()
help(Person)













