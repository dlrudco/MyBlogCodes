import time
from functools import wraps, lru_cache

def logger(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        """wrapper documentation"""
        print(f"----- {function.__name__}: start -----")
        output = function(*args, **kwargs)
        print(f"----- {function.__name__}: end -----")
        return output
    return wrapper

import random
@logger
def say_hello(n):
    """hello documentation"""
    print('hello')
    time.sleep(1)
    return random.random()
