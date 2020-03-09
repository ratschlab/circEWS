
def is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    else:
        return True
