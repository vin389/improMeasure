# From colon-separate integer range to list of integer 
# Range is in Matlab style, not Python style.
# That is, "2:5" mean 2 3 4 5,　not 2 3 4. 
# For example, colonRangeToIntList("2:5  10  15  17:20") would be [2, 3, 4, 5, 10, 15, 17, 18, 19, 20]
# Every character other than number, colon, and space would be replaced by a space character.
def colonRangeToIntList(theStr):
    sequences = []
    # Replace every character other than number, colon, and space by a space character
    theStr = ''.join([c if c.isdigit() or c == ':' or c == ' ' else ' ' for c in theStr])
    # Convert theStr to a list of integers
    for part in theStr.split():
        try:
            # Try converting the part to an integer (single number)
            num = int(part)
            sequences.append(num)
        except ValueError:
            # If conversion fails, assume it's a colon-separated range
            start, end = map(int, part.split(":"))
            sequences.extend(range(start, end+1))
    return sequences
