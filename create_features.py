import numpy as np
import pandas as pd
import csv

#for creates a binary feature which tells if the rocket goes up or down
def create_go_up_and_go_down(table):
    goes_up = np.zeros(len(table))
    goes_down = np.zeros(len(table))
    for i in range(len(table)):
        if(does_go_up(table.loc[i])):
            goes_up[i] = 1
        if(does_go_down(table.loc[i])):
            goes_down[i] = 1
    table["goes_up"] = goes_up
    table["goes_down"] = goes_down
    return table


def does_go_up(row):
    for i in range(30):
        if(row["velZ_"+str(i)] > 0):
            return True
    return False

def does_go_down(row):
    for i in range(30):
        if(row["velZ_"+str(i)] < 0):
            return True
    return False



if __name__ == "__main__":
    table = pd.read_csv("train.csv")
    m_table = create_go_up_and_go_down(table)
    m_table.to_csv("yairs_test.csv", na_rep='NaN')
