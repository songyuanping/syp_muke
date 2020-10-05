class check:
    def __init__(self, func):
        self.f = func

    def __call__(self, *args, **kwargs):
        print('登录验证')
        self.f()


@check
def sendout():
    print('发说说')


sendout()
class check1:
    def __init__(self, func):
        self.f = func

    def __call__(self, *args, **kwargs):
        print('登录验证')
        self.f()


# @check
def sendout():
    print('发说说')


p=check1(sendout)
p()
