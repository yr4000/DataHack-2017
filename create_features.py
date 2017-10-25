import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#for creates a binary feature which tells if the rocket goes up or down
def create_go_up_and_go_down(table):
    goes_up = np.zeros(len(table))
    goes_down = np.zeros(len(table))
    for i in range(len(table)):
        if(does_go_up(table.loc[i])):
            goes_up[i] = 1
        if(does_go_down(table.loc[i])):
            goes_down[i] = 1

    return goes_up, goes_down


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


#get parabula parameters
def calc_S(row):
    S = []
    for i in range(30):
        if(np.isnan(row["posX_"+str(i)])):
            break
        S.append(((row["posX_"+str(i)])**2 + (row["posY_"+str(i)])**2)**0.5)
    return S

def calc_parabola_params(table):
    params = np.array([[0 for i in range(len(table))] for j in range(3)])
    for i in range(len(table)):
        row = table.loc[i]
        S = calc_S(row)
        current_params = np.polynomial.polynomial.polyfit([S[i] for i in range(len(S))],
                                    [row["posZ_"+str(i)] for i in range(len(S))],2)
        if(i%1000 == 0):
            plt.plot(S, [row["posZ_"+str(i)] for i in range(len(S))])
            plt.savefig("images/yair_" + str(i) + ".png")
            plt.clf()
        params[0][i], params[1][i], params[2][i] = current_params[0], current_params[1], current_params[2]
    return params


if __name__ == "__main__":
    table = pd.read_csv("train.csv")
    parabola_params = calc_parabola_params(table)
    goes_up, goes_down = create_go_up_and_go_down(table)
    table["goes_up"] = goes_up
    table["goes_down"] = goes_down
    table["parabola_parameter_a"] = parabola_params[0]
    table["parabola_parameter_b"] = parabola_params[1]
    table["parabola_parameter_c"] = parabola_params[2]

    table.to_csv("yairs_test.csv", na_rep='NaN')
