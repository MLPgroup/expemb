import sympy as sp
from zss import simple_distance, distance, Node
from typing import List
from queue import Queue
from .tokenizer import EquivExpTokenizer
from .timeout import timeout


class TreeEditDistanceCalculator:
    TRIG_OPS = ["sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc"]
    HYP_OPS = ["sinh", "cosh", "tanh", "coth", "sech", "csch", "asinh", "acosh", "atanh", "acoth", "asech", "acsch"]
    LOG_EXP_OPS = ["log", "ln", "exp"]


    def __init__(self, allow_same_group_replacement: bool, remove_const_mul_add: bool, simplify: bool) -> None:
        super(TreeEditDistanceCalculator, self).__init__()
        self.tokenizer = EquivExpTokenizer()
        self.allow_same_group_replacement = allow_same_group_replacement
        self.remove_const_mul_add = remove_const_mul_add
        self.simplify = simplify


    def strdist(self, a: str, b: str):
        return 0 if a == b else 1


    def get_op_group(self, op: str):
        if op in TreeEditDistanceCalculator.TRIG_OPS:
            return "0"
        elif op in TreeEditDistanceCalculator.HYP_OPS:
            return "1"
        elif op in TreeEditDistanceCalculator.LOG_EXP_OPS:
            return "2"
        else:
            return "-1"


    def update_cost(self, a: str, b: str):
        if not self.allow_same_group_replacement:
            return self.strdist(a, b)
        else:
            group_a = self.get_op_group(a)
            group_b = self.get_op_group(b)
            if group_a == "-1" or group_b == "-1":
                return self.strdist(a, b)
            else:
                return self.strdist(group_a, group_b)


    def parse_int(self, token_list: List[str]):
        output = []
        idx = 0
        while idx < len(token_list):
            token = token_list[idx]
            if not token in ["INT", "INT+", "INT-"]:
                output.append(token)
                idx += 1
                continue

            int_val, l = self.tokenizer.parse_int(token_list[idx:])
            output.append(str(int_val))
            idx += l

        return output


    @timeout(2)
    def _filter_constants(self, sympy_exp):
        cls = type(sympy_exp)
        is_add = isinstance(sympy_exp, sp.Add)
        is_mul = isinstance(sympy_exp, sp.Mul)
        args = sympy_exp.args

        if len(args) == 0:
            return sympy_exp

        new_args = []
        for arg in args:
            if is_add or is_mul:
                if not arg.is_constant():
                    new_args.append(self._filter_constants(arg))
                else:
                    new_args.append(sp.S("1") if is_mul else sp.S("0"))
            else:
                new_args.append(self._filter_constants(arg))

        assert len(new_args) > 0, f"No arguments left after constant replacement"

        sympy_exp = cls(*new_args)
        return sympy_exp



    def is_constant(self, sympy_exp):
        @timeout(2)
        def _is_constant(sympy_exp):
            return sympy_exp.is_constant()

        try:
            return _is_constant(sympy_exp)
        except:
            return False


    def filter_constants(self, sympy_exp):
        if self.is_constant(sympy_exp):
            filtered_sympy_exp = sympy_exp
        else:
            filtered_sympy_exp = self._filter_constants(sympy_exp)

        return self.tokenizer.sympy_to_prefix(filtered_sympy_exp)


    def convert_to_tree(self, prefix_exp: str):
        token_list = prefix_exp.split(" ")
        token_list = self.parse_int(token_list)

        stack = []
        for token in reversed(token_list):
            is_op = token in self.tokenizer.get_operators()

            if not is_op:
                node = Node(token, [])
            else:
                n_children = self.tokenizer.get_operators()[token]
                assert len(stack) >= n_children

                children = []
                for _ in range(n_children):
                    children.append(stack.pop())

                # sort children if operator is not commutative.
                # if n_children > 1 and token not in ["sub", "pow", "div", "rac"]:
                #     children.sort(key = lambda x : self.tokenizer.get_index(x.label) if x.label in self.tokenizer.operators else 0)

                node = Node(token, children)

            stack.append(node)

        assert len(stack) == 1, f"prefix_exp: {prefix_exp}, stack: {len(stack)}"

        tree = stack.pop()

        return tree


    def are_equivalent(self, sympy_exp1: str, sympy_exp2: str):
        @timeout(2)
        def _are_equivalent(exp1, exp2):
            return sp.simplify(exp1 - exp2) == 0

        try:
            return _are_equivalent(sympy_exp1, sympy_exp2)
        except:
            return False


    def align_children(self, tree1: Node, tree2: Node):
        queue = Queue()
        queue.put((tree1, tree2))

        while not queue.empty():
            node1, node2 = queue.get()
            node1children, node2children = Node.get_children(node1), Node.get_children(node2)
            node1labels = [Node.get_label(n) for n in node1children]
            node2children.sort(key = lambda node : node1labels.index(Node.get_label(node)) if Node.get_label(node) in node1labels else 2000)

            for n1c, n2c in zip(node1children, node2children):
                queue.put((n1c, n2c))


    def tree_distance(self, prefix_exp1: str, prefix_exp2: str):
        sympy_exp1 = self.tokenizer.prefix_to_sympy(prefix_exp1, evaluate = self.simplify)
        sympy_exp2 = self.tokenizer.prefix_to_sympy(prefix_exp2, evaluate = self.simplify)
        if self.are_equivalent(sympy_exp1, sympy_exp2):
            return 0
        else:
            if self.remove_const_mul_add:
                try:
                    filtered_prefix_exp1 = self.filter_constants(sympy_exp1)
                    filtered_prefix_exp2 = self.filter_constants(sympy_exp2)
                except Exception as e:
                    print(f"Error: {e}. exp1: {prefix_exp1}, exp2: {prefix_exp2}")
                    filtered_prefix_exp1 = prefix_exp1
                    filtered_prefix_exp2 = prefix_exp2
            else:
                filtered_prefix_exp1 = prefix_exp1
                filtered_prefix_exp2 = prefix_exp2

            tree1 = self.convert_to_tree(filtered_prefix_exp1)
            tree2 = self.convert_to_tree(filtered_prefix_exp2)

            self.align_children(tree1, tree2)

            return distance(
                tree1, tree2, Node.get_children,
                insert_cost = lambda node: self.strdist('', Node.get_label(node)),
                remove_cost = lambda node: self.strdist(Node.get_label(node), ''),
                update_cost = lambda a, b: self.update_cost(Node.get_label(a), Node.get_label(b)),
                return_operations = False,
            )
