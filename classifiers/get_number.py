
def is_number_or_dot(char):
    response = (char == "1" or char == "2" or char == "3" or
                char == "4" or char == "5" or char == "6" or
                char == "7" or char == "8" or char == "9" or
                char == "0" or char == ".")
    return response


def get_last_number_pos(string, start):
    """
    If it's not a number, return is -2
    """
    if (not is_number_or_dot(string[start])):
        return -2
    i = start
    while(is_number_or_dot(string[i+1])):
        i += 1;
    return i