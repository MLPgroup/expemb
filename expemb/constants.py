import sympy as sp
from collections import OrderedDict

# Credits: https://github.com/facebookresearch/SymbolicMathematics/blob/main/src/envs/char_sp.py

#################################
######## ExpEmb datasets ########
#################################
OPERATORS = {
    # Elementary functions
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "rac": 2,
    "inv": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "sqrt": 1,
    "exp": 1,
    "ln": 1,
    "abs": 1,
    "sign": 1,
    # Trigonometric Functions
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "cot": 1,
    "sec": 1,
    "csc": 1,
    # Trigonometric Inverses
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "acot": 1,
    "asec": 1,
    "acsc": 1,
    # Hyperbolic Functions
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "coth": 1,
    "sech": 1,
    "csch": 1,
    # Hyperbolic Inverses
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "acoth": 1,
    "asech": 1,
    "acsch": 1,
}
CONSTANTS = ["pi", "E"]
VARIABLES = OrderedDict({
    "x": sp.Symbol("x", real=True, nonzero=True),
})
SYMBOLS = ["I", "INT+", "INT-", "INT", "FLOAT", "-", ".", "10^", "Y"]
SYMPY_OPERATORS = {
    # Elementary functions
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    sp.sign: "sign",
    # Trigonometric Functions
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    sp.sec: "sec",
    sp.csc: "csc",
    # Trigonometric Inverses
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    sp.asec: "asec",
    sp.acsc: "acsc",
    # Hyperbolic Functions
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
    sp.sech: "sech",
    sp.csch: "csch",
    # Hyperbolic Inverses
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
}
SPECIAL_WORDS = ["SOE", "EOE", "PAD"]
INT_BASE = 10
COEFFICIENTS = OrderedDict({
    f'a{i}': sp.Symbol(f'a{i}', real=True)
    for i in range(10)
})


#################################
######## SemVec datasets ########
#################################
SEMVEC_OPERATORS = {
    # Logical Opeations:
    "and": 2,
    "or": 2,
    "not": 1,
    "xor": 2,
    "implies": 2,
    "add": 2,
    "multiply": 2,
    "subtract": 2,
}
SEMVEC_CONSTANTS = ["true", "false"]
SEMVEC_VARIABLES = OrderedDict({
    "a": sp.Symbol("a", real=True),
    "c": sp.Symbol("c", real=True),
    "b": sp.Symbol("b", real=True),
    "d": sp.Symbol("d", real=True),
    "e": sp.Symbol("e", real=True),
    "f": sp.Symbol("f", real=True),
    "g": sp.Symbol("g", real=True),
    "h": sp.Symbol("h", real=True),
    "i": sp.Symbol("i", real=True),
    "j": sp.Symbol("j", real=True),
})
SEMVEC_SYMPY_OPERATORS = {
    # Logical Opeations
    sp.And: "and",
    sp.Or: "or",
    sp.Not: "not",
    sp.Xor: "xor",
    sp.Implies: "implies",
    sp.Add: "add",
    sp.Mul: "multiply",
    sp.Add: "subtract",
}
