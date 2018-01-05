from inspect import signature
from sklearn.base import BaseEstimator


class Foo:
    def __init__(self, arg1, arg2, **args):
        self.arg1 = arg1
        self.arg2 = arg2
        self.args = args

    def __str__(self):
        return "arg1: {}\narg2: {}\nargs: {}".format(self.arg1, self.arg2, self.args)

class Bar(BaseEstimator):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return "arg1: {}\narg2: {}".format(self.arg1, self.arg2)


if __name__ == '__main__':
    # f = Foo("A", 1, x=2, y=3)
    # print(f)

    # init = getattr(Foo.__init__, 'deprecated_original', Foo.__init__)
    #
    # init_signature = signature(init)
    # # Consider the constructor parameters excluding 'self'
    # # parameters = [p for p in init_signature.parameters.values()
    # #               if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    #
    # parameters = [p for p in init_signature.parameters.values()]
    #
    # print(parameters)

    b = Bar("A", 1)
    print(b)
    b.set_params(arg1="B", arg2=2)
    print(b)
