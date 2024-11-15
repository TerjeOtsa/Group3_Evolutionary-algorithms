# Calculate fuel consumption per vehicle based on speed.
def fuel_consumption_model(speed, a=0.01, b=2, c=0.1):
    # speed: Average speed in mph.
    # a: Constant for linear dependence on speed.
    # b:  Constant for dependence on inverse of speed.
    # c : Constant for base fuel consumption.
    return a * speed + b / speed + c

# Calculate total fuel consumption for a road segment
def total_fuel_consumption(volume, speed, segment_length, a=0.01, b=2, c=0.1):
    fc_per_vehicle = fuel_consumption_model(speed, a, b, c)
    return int(volume * fc_per_vehicle * segment_length)    # Returns total fuel consumption. This output is given in FUEL UNITS, as our given fuel consumption model does not specify what spesific units we get.
                                                            # Volume however is per vehicle, Segment length is given in miles, and speed is given in miles per hour
