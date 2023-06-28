class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # y = creator(x)
        f = self.creator
        if f is not None:
            x = f.input
            # dx = df(x)*gy
            x.grad = f.backward(self.grad)
            # do the next grad
            x.backward()


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
