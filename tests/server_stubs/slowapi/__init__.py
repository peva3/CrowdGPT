class Limiter:
    def __init__(self, *args, **kwargs):
        pass
    def limit(self, *args, **kwargs):
        return lambda fn: fn

def _rate_limit_exceeded_handler(*args, **kwargs):
    return None
