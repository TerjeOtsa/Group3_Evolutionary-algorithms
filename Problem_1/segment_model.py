import re

def calculate_block(from_street, to_street):
    # Extracting numbers using regular expressions
    from_number = int(re.search(r'\d+', from_street).group())
    to_number = int(re.search(r'\d+', to_street).group())
    return abs(from_number - to_number)


# Segment lenght calculations, based of average block length in New York City
def calculate_segment(block_number):
    average_block_lenght = 80.4
    return block_number * average_block_lenght


# Example usage
from_street = "87th STREET"
to_street = "88 STREET"
print(calculate_block(from_street, to_street))  # Output should be 1, if numbers are consecutive
