class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # y = creator(x)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            # dx = df(x)*dy
            x.grad = f.backward(f.grad)
            # do the next grad
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError
