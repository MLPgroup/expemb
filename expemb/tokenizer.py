import torch
import sympy as sp
from torch import Tensor
from abc import ABC, abstractmethod
from .constants import *


class Tokenizer(ABC):
    @abstractmethod
    def get_index(self, comp: str) -> int:
        pass


    @abstractmethod
    def get_comp(self, idx: int) -> str:
        pass


    @abstractmethod
    def get_operators(self) -> dict:
        pass


    @abstractmethod
    def get_variables(self) -> dict:
        pass


    @abstractmethod
    def get_coefficients(self) -> dict:
        pass


    @abstractmethod
    def get_constants(self) -> dict:
        pass


    @abstractmethod
    def get_sympy_operators(self) -> dict:
        pass


    def get_pad_token(self) -> str:
        return "PAD"


    def get_pad_index(self) -> int:
        return self.get_index(self.get_pad_token())


    def get_soe_token(self) -> str:
        return "SOE"


    def get_soe_index(self) -> int:
        return self.get_index(self.get_soe_token())


    def get_eoe_token(self) -> str:
        return "EOE"


    def get_eoe_index(self) -> int:
        return self.get_index(self.get_eoe_token())


    def encode(self, exp: str) -> Tensor:
        indexes = [self.get_index("SOE")] + [self.get_index(comp) for comp in exp.split(" ")] + [self.get_index("EOE")]
        return torch.LongTensor(indexes).view(-1)


    def decode(self, encoded: Tensor, ignore_after_eoe: bool = False) -> str:
        assert encoded.ndim == 1, f"1-D tensors are allowed."
        exp_arr = []
        for elem in encoded:
            comp = self.get_comp(elem.item())
            exp_arr.append(comp)
            if ignore_after_eoe and comp == self.get_eoe_token():
                break

        return " ".join(exp_arr)


    def prefix_to_sympy(self, expr, evaluate=True):
        p, r = self.prefix_to_infix(expr)
        if len(r) > 0:
            raise Exception(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")

        local_dict = self.get_sympy_local_dict()
        expr = sp.parsing.sympy_parser.parse_expr(f'({p})', evaluate=evaluate, local_dict=local_dict)
        return expr


    def get_sympy_local_dict(self) -> dict:
        local_dict = {}
        for k, v in list(self.get_variables().items()) + list(self.get_coefficients().items()):
            assert k not in local_dict
            local_dict[k] = v
        return local_dict


    def prefix_to_infix(self, expr):
        return self._prefix_to_infix(expr.split(" "))


    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
            - infix mode (returns human readable string)
            - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise Exception("Empty prefix list.")
        t = expr[0]

        if t in self.get_operators():
            args = []
            l1 = expr[1:]
            for _ in range(self.get_operators()[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.get_variables() or t in self.get_coefficients() or t in self.get_constants() or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]


    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = 10
        balanced = False
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise Exception(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1


    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub' or token == 'subtract':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul' or token == 'multiply':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'and':
            return f'({args[0]})&({args[1]})'
        elif token == 'or':
            return f'({args[0]})|({args[1]})'
        elif token == 'xor':
            return f'({args[0]})^({args[1]})'
        elif token == 'implies':
            return f'({args[0]})>>({args[1]})'
        elif token == 'not':
            return f'~({args[0]})'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token


    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.get_variables() and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix_helper(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list

        assert (op == 'add' or op == 'mul') and (n_args >= 2) \
            or (op == 'not') and (n_args == 1) \
            or (op == 'and' or op == 'or' or op == 'xor') and (n_args >= 2) \
            or (op != 'add' and op != 'mul' and op != 'and' or op != 'or' or op != 'xor') and (1 <= n_args <= 2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix_helper(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix_helper(expr.args[i])

        return parse_list


    def sympy_to_prefix(self, expr):
        return " ".join(self.sympy_to_prefix_helper(expr))


    def sympy_to_prefix_helper(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        elif expr == sp.false:
            return ['false']
        elif expr == sp.true:
            return ['true']
        # SymPy operator
        for op_type, op_name in self.get_sympy_operators().items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise Exception(f"Unknown SymPy operator: {expr}")


    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = 10
        balanced = False
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')
        return res[::-1]


class EquivExpTokenizer(Tokenizer):
    def __init__(self):
        super(EquivExpTokenizer, self).__init__()
        self.special_words = SPECIAL_WORDS
        self.int_base = INT_BASE
        self.operators = OPERATORS
        self.coefficients = COEFFICIENTS
        self.variables = VARIABLES
        self.symbols = SYMBOLS
        self.constants = CONSTANTS
        self.sympy_operators = SYMPY_OPERATORS
        self.elements = [str(i) for i in range(abs(INT_BASE))]

        self.components = self.special_words + self.constants + list(self.variables.keys()) + list(self.operators.keys()) + self.symbols + self.elements
        self.comp2index = {comp : idx for idx, comp in enumerate(self.components)}
        self.index2comp = {idx : comp for comp, idx in self.comp2index.items()}
        self.n_comp = len(self.comp2index)


    def get_index(self, comp: str) -> int:
        return self.comp2index[comp]


    def get_comp(self, idx: int) -> str:
        return self.index2comp[idx]


    def get_operators(self) -> dict:
        return self.operators


    def get_variables(self) -> dict:
        return self.variables


    def get_coefficients(self) -> dict:
        return self.coefficients


    def get_constants(self) -> dict:
        return self.constants


    def get_sympy_operators(self) -> dict:
        return self.sympy_operators


class SemVecTokenizer(Tokenizer):
    def __init__(self):
        super(SemVecTokenizer, self).__init__()
        self.special_words = SPECIAL_WORDS
        self.operators = SEMVEC_OPERATORS
        self.variables = SEMVEC_VARIABLES
        self.constants = SEMVEC_CONSTANTS
        self.sympy_operators = SEMVEC_SYMPY_OPERATORS
        self.components = self.special_words + self.constants + list(self.variables.keys()) + list(self.operators.keys())
        self.comp2index = {comp : idx for idx, comp in enumerate(self.components)}
        self.index2comp = {idx : comp for comp, idx in self.comp2index.items()}
        self.n_comp = len(self.comp2index)


    def get_index(self, comp: str) -> int:
        return self.comp2index[comp]


    def get_comp(self, idx: int) -> str:
        return self.index2comp[idx]


    def get_operators(self) -> dict:
        return self.operators


    def get_variables(self) -> dict:
        return self.variables


    def get_coefficients(self) -> dict:
        return {}


    def get_constants(self) -> dict:
        return self.constants


    def get_sympy_operators(self) -> dict:
        return self.sympy_operators
