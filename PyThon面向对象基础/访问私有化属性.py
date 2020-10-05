class Person(object):
    def __init__(self):
        __age=18
    def setAge(self,value):
        if isinstance(value,int) and 0<value<200:
            self.__age=value
        else:
            print('你输入的数据有问题，请重新输入！')
    def getAge(self):
        return self.__age
p=Person()
p.setAge(220)
print(p.getAge())

