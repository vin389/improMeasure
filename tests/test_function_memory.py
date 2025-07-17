def foo(a=None, b=None):
    if a is None:
        a = 0
    if b is None:
        b = 0
    return a + b

c = foo(1, 2)

d = foo()

print(c, d)
