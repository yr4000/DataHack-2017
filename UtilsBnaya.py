import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


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

def get_stats(predictions,test_factorized):
    right = 0
    wrong = 0

    for i in range(0,len(predictions)):
        if test_factorized[i] == predictions[i]:
            right +=1
        else:
            wrong+=1
    return [right,wrong]



def run_random_forest(df,label_name):
    """
        @Returns: numpy array with predictions.
        @Prints: how many times model was right/wrong.
        @Param: DataFrame.
    """
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.80
    train, test = df[df['is_train']==True], df[df['is_train']==False]
    labels = df[label_name]
    features = df.drop(label_name,axis=1)

    train_factorized = pd.factorize(train[label_name])[0] #Convert label names to factor.
    test_factorized = pd.factorize(test[label_name])[0]

    #len_test = len(test_factorized)

    rf_classifier = RandomForestClassifier(n_jobs = 2, random_state=0)
    rf_classifier.fit(train[features], train_factorized)

    predictions = rf_classifier.predict(test[features])

    print get_stats(predictions,test_factorized)
    print len(test_factorized)
    return predictions



