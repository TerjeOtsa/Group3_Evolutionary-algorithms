def fuel_consumption_model(speed, a=0.01, b=2, c=0.1):
    """
    Calculate fuel consumption per vehicle based on speed.
    
    Parameters:
    speed (float): Average speed in mph.
    a (float): Constant for linear dependence on speed.
    b (float): Constant for dependence on inverse of speed.
    c (float): Constant for base fuel consumption.
    
    Returns:
    float: Fuel consumption per vehicle.
    """

    if speed <= 0:
        # print("Warning: Speed is zero or negative, adjusting to avoid division by zero.")
        speed = 0.1  # Adjust to a minimum positive speed to avoid division by zero
    return a * speed + b / speed + c

def total_fuel_consumption(volume, speed, segment_length, a=0.01, b=2, c=0.1):
    """
    Calculate total fuel consumption for a road segment and time interval.
    
    Parameters:
    volume (float): Vehicle count in the time interval.
    speed (float): Average speed in the time interval (mph).
    segment_length (float): Length of the road segment (miles).
    a (float): Constant for linear dependence on speed.
    b (float): Constant for dependence on inverse of speed.
    c (float): Constant for base fuel consumption.
    
    Returns:
    float: Total fuel consumption for the time interval.
    """
    # Fuel consumption per vehicle
    fc_per_vehicle = fuel_consumption_model(speed, a, b, c)
    
    # Total fuel consumption
    return int(volume * fc_per_vehicle * segment_length)