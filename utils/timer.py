import time 


def timer(func):
    """A decorator that prints how long a function took to run.

    Args:
        func (callable): The function or class being decorated.
    
    Returns:
        callable: The decorated function or class.

    Examples:
        from lib.utils import timer

        
        @timer()
        def func(args)

        >>> func()
        >>> ====='[func name]' took [elapsed time in seconds] s.=====

    Note:
        You should import `timer` like the following,
            from lib.utils import timer 

    """
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)

        t_total = time.time() - t_start 
        print(f"====='{func.__name__}' took {t_total:.5f} s.=====")
        return result 
    return wrapper