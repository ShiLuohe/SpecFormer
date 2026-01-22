import torch
from functools import wraps

def possible_no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        no_grad = kwargs.get("no_grad", False)
        try: kwargs.pop("no_grad")
        except: pass
        if no_grad:
            with torch.no_grad():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper