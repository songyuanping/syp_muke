from win32com import client


class Caculator:
    def __say(self, word):
        # 1.创建一个语音播报对象
        speaker = client.Dispatch("SAPI.SpVoice")
        # 2.通过这个播报对象直接播报出相应的语音字符串即可
        speaker.Speak(word)

    # 装饰器模式
    def __check_num(func):
        def inner(self, n):
            if not isinstance(n, int):
                raise TypeError("当前这个数据类型有问题，应该是一个整形数据！")
            return func(self, n)

        return inner

    def __create_say(word=""):
        def __say_num(func):
            def inner(self, n):
                self.__say(word + str(n))
                return func(self, n)

            return inner

        return __say_num

    # 先执行check_num，后执行say_num
    @__check_num
    @__create_say()
    def __init__(self, num=1):
        self.__result = num

    @__check_num
    @__create_say('+')
    def plus(self, num):
        self.__result += num
        return self

    @__check_num
    @__create_say('-')
    def sub(self, num):
        self.__result -= num
        return self

    @__check_num
    @__create_say('*')
    def mul(self, num):
        self.__result *= num
        return self

    @__check_num
    @__create_say("/")
    def div(self, num):
        self.__result /= num
        return self
    def clear(self):
        self.__result=0
        return self
    def show(self):
        self.__say('the result is {}'.format(self.__result))
        print('result is {}'.format(self.__result))
        return self

    @property
    def result(self):
        return self.__result


c1 = Caculator(6)
c1.plus(3).sub(2).mul(17).div(4).show().clear().plus(777).sub(123).show()
print(c1.result)
