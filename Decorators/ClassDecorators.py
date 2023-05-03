from typing import Dict, Any

from decorator import decorator
from dataclasses import dataclass
# This might be moving a lot of memory between CPU and GPU. I don't necessarily think so, but I'm not sure. It does seem to be running very slowly .

@dataclass
class ScopedRetValue:
    value: Any 
    scope: Dict

def print_func_args_ret(cls_name):
    def make_decorated(func):
        def decorated_func(*args, **kwargs):
            ret = func(*args, **kwargs)
            print(f"{cls_name}.{func.__name__}({args=}, {kwargs=}) = {ret}")
            return ret
        return decorated_func
    return make_decorated

def __merge_scopes(old_scope: Dict, new_scope: Dict):
    # Write new scope values into existing scope
    for k, v in new_scope.items():
        old_scope[k] = v
    return old_scope 

def extendable(returns_scope: bool):
    def make_decorated(func):
        def decorated_func(*args, **kwargs):
            existing_scope = kwargs.get("scope", {})
            if returns_scope is True: 
                scoped_value = func(*args, **kwargs)
                if not isinstance(scoped_value, ScopedRetValue): 
                    raise RuntimeError("Extended function does not return ScopedRetValue") 
                new_scope = __merge_scopes(existing_scope, scoped_value.scope)
                return ScopedRetValue(value = scoped_value.value, scope = new_scope)
            else: 
                return ScopedRetValue(value = func(*args, **kwargs), scope = existing_scope)
        return decorated_func
    return make_decorated

def extend_super(cls, propogate_scope = True):
    def make_decorated(func):
        # TODO: Check if cls.func is extendable. Raise runtime error if not. 
        # TODO: Check if cls is base of class. Raise runtime error if not.  
        def decorated_func(*args, **kwargs): 
            if len(args) < 1: 
                raise TypeError("extend_super(func). Signature must be func(self, ...). Missing arguments")

            # Get obj
            elif len(args) == 1:
                self = args[0]
                args = tuple()
            else: 
                self, *args = args

            super_func = getattr(cls, func.__name__)
            scoped_ret_super = super_func(self, *args, **kwargs)
            if not isinstance(scoped_ret_super, ScopedRetValue):
                raise RuntimeError("Extended function does not return ScopedRetValue") 

            if scoped_ret_super.value is None:
                kwargs["scope"] = scoped_ret_super.scope
                func_ret = func(self, *args, **kwargs)
                if isinstance(func_ret, ScopedRetValue):
                    scoped_ret = ScopedRetValue(value = func_ret.value, scope = __merge_scopes(scoped_ret_super.scope, func_ret.scope))
                else: 
                    scoped_ret = ScopedRetValue(value = func_ret, scope = scoped_ret_super.scope)  

                if propogate_scope is True: 
                    return scoped_ret
                else: 
                    return scoped_ret.value
            else: 
                if propogate_scope is True: 
                    return scoped_ret_super
                else: 
                    return scoped_ret_super.value

        return decorated_func
    return make_decorated

if __name__ == "__main__":

    class A:
        def __init__(self):
            self.calls = 0

        @print_func_args_ret("A")
        @extendable(returns_scope = True)
        def f(self):
            self.calls += 1
            scope = dict(value = 1)
            return ScopedRetValue(value = None, scope = scope)

    class B(A): 
        def __init__(self):
            super().__init__()

        @print_func_args_ret("B")
        @extendable(returns_scope = True)
        @extend_super(A, propogate_scope = True)
        def f(self, scope: Dict = None):
            print(f"b.f():{scope=}")
            scope["value"] += 1
            return ScopedRetValue(value = None, scope = scope)

    class C(B): 
        def __init__(self):
            super().__init__()

        @print_func_args_ret("C")
        @extend_super(B, propogate_scope = False)
        def f(self, scope: Dict = None):
            print(f"c.f():{scope=}")
            return scope["value"] + 1

    a = A()
    aa = A()
    b = B()
    c = C()
    sup = super(type(b), b)


    print("f(): ", a.f(), b.f(), c.f(), c.f())
    # print(a.f())
    # print(b.f())
    # # breakpoint()
    # print(c.f())
    # print(c.f())
    print("f calls: ", [x.calls for x in [a,aa,b,c]])
