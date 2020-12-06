import multipledispatch

__all__ = [
    'area',
    'is_intersect',
    'intersection',
    'union',
]


@multipledispatch.dispatch(float, float, float, float)
def area(x1, y1, x2, y2) -> float:
    return abs((x2-x1) * (y2-y1))

@multipledispatch.dispatch(tuple)
def area(box) -> float:
    if len(box) != 4:
        raise RuntimeError("expect (x1, y1, x2, y2")
    return area(*box)


@multipledispatch.dispatch(*[float]*8)
def is_intersect(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
    return x1a < x2b and x2a > x1b and y1a < y2b and y2a > y1b

@multipledispatch.dispatch(tuple, tuple)
def is_intersect(box_a, box_b):
    if (len(box_a) != 4) or (len(box_b) != 4):
        raise RuntimeError("expect (x1,y1,x2,y2)")
    return is_intersect(*box_a, *box_b)


@multipledispatch.dispatch(*[float]*8)
def intersection(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) -> float:
    if not is_intersect(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
        return 0
    w = min(x2a, x2b) - max(x1a, x1b)
    h = min(y2a, y2b) - max(y1a, y1b)
    return w * h

@multipledispatch.dispatch(tuple, tuple)
def intersection(box_a, box_b) -> float:
    if (len(box_a) != 4) or (len(box_b) != 4):
        raise RuntimeError("expect (x1,y1,x2,y2)")
    return intersection(*box_a, *box_b)


@multipledispatch.dispatch(*[float]*8)
def union(*args):
    area_a, area_b = area(args[:4]), area(args[4:])
    area_ab = area_a + area_b
    area_ab -= intersection(*args)
    return area_ab

@multipledispatch.dispatch(tuple, tuple)
def union(box_a, box_b) -> float:
    if (len(box_a) != 4) or (len(box_b) != 4):
        raise RuntimeError("expect (x1,y1,x2,y2)")
    return union(*box_a, *box_b)
