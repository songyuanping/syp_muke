class Animal:
    _y =66
    __x = 10

    def test(self):
        print(Animal.__x)
        print(self.__x)


class Dog(Animal):
    def test(self):
        print(Animal._y)
        print(self._y)

    def test2(self):
        # print(Dog.__x)
        # print(self.__x)
        pass


print(Animal._y)
# 私有属性，在类外部访问不了
# print(Animal.__x)
print(Animal.__dict__)
print(Animal._Animal__x)

a=Animal()
d=Dog()
a.test()
d.test()
d.test2()