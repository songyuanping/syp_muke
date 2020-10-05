# class Money:
#     pass
#
#
# print(Money.__name__)
# xxx=Money
# print(xxx.__class__)
# # Money=666
# # print(Money)
# one=Money()
# print(one.__class__)
# class Person:
#     pass
# p=Person()
# p.age=18
# p.name='jack'
# # 删除变量和类属性用delete方法
# del p.age
# print(p.__dict__)
# Money.count=1
# print(Money.count)
# print(Money.__dict__)

class Person:
    age = 19
    num = 999
    name = '小明'


# 类的属性不支持如下方式修改，需要通过setattr进行修改
# Person.__dict__['age']=100
p = Person()
p.age = 100
print(p.age, p.num, p.name)
del Person.age
print(p.__dict__, p.name, p.num, p.age)
print(Person.__dict__)
# 对象的属性支持修改
p.__dict__ = {'sex': 'man', 'id': 1234}
print(p.sex, p.id)


class Per:
    sex='M'
    __slots__ = ['age', 'num','name']
    pass


p1 = Per()
p1.age = 1
p1.num = 'min'
print(p1.age,p1.num)
p2 = Per()

p2.name = '123'
# print(p2.__dict__)
