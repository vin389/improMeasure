def is_float(s):
    if type(s) == str:
        s = s.replace('np.nan', 'nan')
        s = s.replace('numpy.nan', 'nan')
    try:
        float(s)
        return True
    except ValueError:
        return False