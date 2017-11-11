# Method: Used to check if two sliding windows overlap
def is_overlap_between_windows(win_a, win_b, win_size):
    """
    :param win_a: Sliding window a
    :param win_b: Sliding window b
    :param win_size: Sliding window size
    :return: True if overlap, else False
    """
    if win_a[0] > (win_b[0]+win_size[0]) or win_b[0] > (win_a[0]+win_size[0]):
        return False
    if win_a[1] > (win_b[1]+win_size[1]) or win_b[1] > (win_a[1]+win_size[1]):
        return False
    return True


# Method: Used to return the region of overlap between sliding windows
def get_overlapping_region(win_a, win_b, win_size):
    """
    :param win_a: Sliding window a
    :param win_b: Sliding window b
    :param win_size: Sliding window size
    :return: Overlapping region
    """
    if not is_overlap_between_windows(win_a, win_b, win_size):
        return None, None, None

    x, y = max(win_a[0], win_b[0]), max(win_a[1], win_b[1])
    size = (min(win_a[0]+win_size[0], win_b[0]+win_size[0]) - x,
            min(win_a[1]+win_size[1], win_b[1]+win_size[1]) - y)
    return x, y, size


win_a = (0, 0)
win_b = (201, 0)
win_size = (200, 200)

res = is_overlap_between_windows(win_a, win_b, win_size)
print(res)
x, y, size = get_overlapping_region(win_a, win_b, win_size)
print(x, y)
print(size)
