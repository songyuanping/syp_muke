class A(object):
    pass
class B(A):
    pass
class C(A):
    pass
class D(B,C):
    pass


print(D.mro())
print(D.__mro__)