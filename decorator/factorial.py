import time
from functools import wraps, lru_cache

def time_tester(function, n=10):
    @wraps(function)
    def wrapper(*args, **kwargs):
        """tester documentation"""
        print(f"----- {function.__name__}: start -----")
        elapsed = 0
        for _ in range(n):
            st = time.time()
            output = function(*args, **kwargs)
            elapsed += time.time()-st
        print(f"----- {function.__name__}: end -----")
        print(f"Elapsed : {elapsed:0.5f}sec")
        return output
    return wrapper

@time_tester
@lru_cache(maxsize=None)
def factorial(n):
    """factorial documentation"""
    answer = 1
    for k in range(1, n+1):
        answer *= k
    return answer
