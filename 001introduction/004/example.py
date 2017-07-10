class F(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * x + self.b


f = F(2.0, -1.0)
print(f(1.0))  # 1.0
print(f(2.0))  # 3.0
