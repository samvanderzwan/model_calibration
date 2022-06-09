# simple class to store measurement data


class MeasData:
    def __init__(self, location, prop, time, values):
        self.location = str(location)
        self.property = prop
        assert len(time) == len(values), "Length time and values is not the same"
        self.time = time
        self.values = values


class Parameter:
    def __init__(self, location, prop, minimum, maximum, iskeyword=False):
        self.location = location
        self.iskeyword = iskeyword
        self.property = prop
        self.minimum = minimum
        self.maximum = maximum
        self.wanda_props = []
