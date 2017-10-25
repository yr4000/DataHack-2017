import numpy

def get_total_time(line):
    """ Funcion for internal use."""
    line = line.split(',')
    time_list = []

    for j in range(8,210,7):
        time_list.append(line[j])

    total_time = 0
    for single_time in time_list:
        if single_time == "NaN":
            break
        else:
            total_time = single_time

    return total_time



def get_time_list(csv_file_path):
    """
        Returns numpy array with the travel time of every missle.
        @Param: CSV file path.
    """
    time_list = []
    with open(csv_file_path, 'r') as csv_file_obj:
        csv_content = csv_file_obj.readlines()
        first_line = True

        for line in csv_content:
            if first_line:
                first_line = False
                continue
            else:
                time_list.append(get_total_time(line))
        return  numpy.array(time_list)


