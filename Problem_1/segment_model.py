import re

def calculate_block(from_street, to_street, borough):
    # Use specified lengths for Staten Island and Brooklyn. 
    # This is because the dataset does not have numbers in From and To columns across the whole volume dataset as I could find.
    if borough == "Staten Island":
        return 8                    # Based of miles from google maps, can be seen in Queries folder
    elif borough == "Brooklyn":
        return 10                   # Based of miles from google maps

    # For other boroughs, calculate based on numbers in 'from_street' and 'to_street'
    from_match = re.search(r'\d+', from_street)
    to_match = re.search(r'\d+', to_street)
    
    # Check if both segments have numbers, calculate the difference if so
    if from_match and to_match:
        from_number = int(from_match.group())
        to_number = int(to_match.group())
        return abs(from_number - to_number)
    else:
        return 1  # Default length if no numbers are present


# Segment lenght calculations, based of average block length in New York City
def calculate_segment(block_number):
    average_block_lenght = 0.05     # miles
    return block_number * average_block_lenght


# # Example usage
# from_street = "87th STREET"
# to_street = "88 STREET"
# print(calculate_block(from_street, to_street, "Brooklyn"))  # Output should be 1, if numbers are consecutive
