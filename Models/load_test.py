from sympy_to_c import sympy_to_c as sp2c
dx = sp2c.load_func_from_solib('cartPole.so', 'cartPole.c', 5)