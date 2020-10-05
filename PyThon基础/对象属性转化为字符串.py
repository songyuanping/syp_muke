class Person:
    def __init__(self,age=18,name='roam'):
        self.age=age
        self.name=name
    def __str__(self):
        return "str name:%s age:%s"%(self.name,self.age)
    # repr函数主要提供给开发人员使用
    def __repr__(self):
        return "repr name:%s age %s"%(self.name,self.age)
p1=Person()
print(p1)
p2=Person(19,'rose')
print(p2)
print(repr(p1))
class PenFactory:
    def __init__(self,p_type):
        self.p_type=p_type
    def __call__(self, p_color):
        print("创建了一个%s类型的画笔，它是%s颜色！"%(self.p_type,p_color))
penF=PenFactory("钢笔")
penF('蓝色')
penF('黄色')
penF('绿色')
pencilF=PenFactory('铅笔')
pencilF('蓝色')
pencilF('黄色')
pencilF('绿色')


