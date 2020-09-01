from functools import wraps

def augmentation_label_handler(augmentation_fn):
    """ deals with functions that also return a label, by augmenting the data,
    but not the label (if present) """
    @wraps(augmentation_fn)
    def _wrapper(*args, **kwargs):
        if isinstance(args, tuple):
            data, label = args
            return augmentation_fn(data, **kwargs), label
        else:
            return augmentation_fn(args, **kwargs)
    return _wrapper
