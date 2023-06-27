class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input 
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError
