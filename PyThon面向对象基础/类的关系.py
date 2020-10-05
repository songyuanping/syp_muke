# <class 'type'>
print(type(int))
# <class 'type'>
print(int.__class__)
# <class 'type'>
print(type(bool))
# <class 'int'>
print(bool.__base__)
# <class 'type'>type自己创建自身
print(type(type))
# <class 'object'>type继承自object
print(type.__base__)
# <class 'type'>
print(type.__class__)
# <class 'type'>object是type的实例化
print(type(object))
# <class 'type'>
print(object.__class__)
# None
print(object.__base__)