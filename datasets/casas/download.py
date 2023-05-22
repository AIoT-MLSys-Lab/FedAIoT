import os
import zipfile

import gdown
import numpy as np
import pandas as pd
import os
import re
from collections import Counter
from datetime import datetime
from tensorflow.keras.utils import pad_sequences
# Define the shared Google Drive file URL
FILE_ID = "1Ni1aJIpazW6dSn64h7P9GKtXvvH-yXqz"

# Define the directory where you want to save the dataset
SAVE_DIR = "./datasets/casas"


# Function to download the file from Google Drive
def download_file_from_google_drive(file_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "casas.zip")
    gdown.download(output=file_path, quiet=False, id=file_id)

    return file_path


# Function to extract the dataset
def extract_file(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Extracted dataset to {save_dir}")


offset = 20
max_lenght = 2000

cookActivities = {"cairo": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Leave_Home": offset + 4,
                            "Eat": offset + 5,
                            "Bed_to_toilet": offset + 6,
                            "Bathing": offset + 7,
                            "Enter_home": offset + 8,
                            "Personal_hygiene": offset + 9,
                            "Relax": offset + 10,
                            "Cook": offset + 11},
                  "twor.2009": {"Other": offset,
                                "Work": offset + 1,
                                "Sleep": offset + 2,
                                "Relax": offset + 3,
                                "Personal_hygiene": offset + 4,
                                "Cook": offset + 5,
                                "Bed_to_toilet": offset + 6,
                                "Bathing": offset + 7,
                                "Eat": offset + 8,
                                "Take_medicine": offset + 9,
                                "Enter_home": offset + 10,
                                "Leave_home": offset + 11},
                  "twor.summer.2009": {"Other": offset,
                                       "Bathing": offset + 1,
                                       "Cook": offset + 2,
                                       "Sleep": offset + 3,
                                       "Work": offset + 4,
                                       "Bed_to_toilet": offset + 5,
                                       "Personal_hygiene": offset + 6,
                                       "Relax": offset + 7,
                                       "Eat": offset + 8,
                                       "Take_medicine": offset + 9,
                                       "Enter_home": offset + 10,
                                       "Leave_home": offset + 11}
    ,
                  "twor.2010": {"Other": offset,
                                "Work": offset + 1,
                                "Sleep": offset + 2,
                                "Relax": offset + 3,
                                "Personal_hygiene": offset + 4,
                                "Leave_Home": offset + 5,
                                "Enter_home": offset + 6,
                                "Eat": offset + 7,
                                "Cook": offset + 8,
                                "Bed_to_toilet": offset + 9,
                                "Bathing": offset + 10,
                                "Take_medicine": offset + 11},
                  "milan": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Relax": offset + 4,
                            "Leave_Home": offset + 5,
                            "Eat": offset + 6,
                            "Cook": offset + 7,
                            "Bed_to_toilet": offset + 8,
                            "Bathing": offset + 9,
                            "Enter_home": offset + 10,
                            "Personal_hygiene": offset + 11},
                  }
mappingActivities = {"cairo": {"": "Other",
                               "R1 wake": "Other",
                               "R2 wake": "Other",
                               "Night wandering": "Other",
                               "R1 work in office": "Work",
                               "Laundry": "Work",
                               "R2 take medicine": "Take_medicine",
                               "R1 sleep": "Sleep",
                               "R2 sleep": "Sleep",
                               "Leave home": "Leave_Home",
                               "Breakfast": "Eat",
                               "Dinner": "Eat",
                               "Lunch": "Eat",
                               "Bed to toilet": "Bed_to_toilet"},
                     "twor.2009": {"R1_Bed_to_Toilet": "Bed_to_toilet",
                                   "R2_Bed_to_Toilet": "Bed_to_toilet",
                                   "Meal_Preparation": "Cook",
                                   "R1_Personal_Hygiene": "Personal_hygiene",
                                   "R2_Personal_Hygiene": "Personal_hygiene",
                                   "Watch_TV": "Relax",
                                   "R1_Sleep": "Sleep",
                                   "R2_Sleep": "Sleep",
                                   "Clean": "Work",
                                   "R1_Work": "Work",
                                   "R2_Work": "Work",
                                   "Study": "Other",
                                   "Wash_Bathtub": "Other",
                                   "": "Other"},
                     "twor.summer.2009": {"R1_shower": "Bathing",
                                          "R2_shower": "Bathing",
                                          "Bed_toilet_transition": "Other",
                                          "Cooking": "Cook",
                                          "R1_sleep": "Sleep",
                                          "R2_sleep": "Sleep",
                                          "Cleaning": "Work",
                                          "R1_work": "Work",
                                          "R2_work": "Work",
                                          "": "Other",
                                          "Grooming": "Other",
                                          "R1_wakeup": "Other",
                                          "R2_wakeup": "Other"},
                     "twor.2010": {"": "Other",
                                   "R1_Wandering_in_room": "Other",
                                   "R2_Wandering_in_room": "Other",
                                   "R1_Work": "Work",
                                   "R2_Work": "Work",
                                   "R1_Housekeeping": "Work",
                                   "R1_Sleeping_Not_in_Bed": "Sleep",
                                   "R2_Sleeping_Not_in_Bed": "Sleep",
                                   "R1_Sleep": "Sleep",
                                   "R2_Sleep": "Sleep",
                                   "R1_Watch_TV": "Relax",
                                   "R2_Watch_TV": "Relax",
                                   "R1_Personal_Hygiene": "Personal_hygiene",
                                   "R2_Personal_Hygiene": "Personal_hygiene",
                                   "R1_Leave_Home": "Leave_Home",
                                   "R2_Leave_Home": "Leave_Home",
                                   "R1_Enter_Home": "Enter_home",
                                   "R2_Enter_Home": "Enter_home",
                                   "R1_Eating": "Eat",
                                   "R2_Eating": "Eat",
                                   "R1_Meal_Preparation": "Cook",
                                   "R2_Meal_Preparation": "Cook",
                                   "R1_Bed_Toilet_Transition": "Bed_to_toilet",
                                   "R2_Bed_Toilet_Transition": "Bed_to_toilet",
                                   "R1_Bathing": "Bathing",
                                   "R2_Bathing": "Bathing"},
                     "milan": {"": "Other",
                               "Master_Bedroom_Activity": "Other",
                               "Meditate": "Other",
                               "Chores": "Work",
                               "Desk_Activity": "Work",
                               "Morning_Meds": "Take_medicine",
                               "Eve_Meds": "Take_medicine",
                               "Sleep": "Sleep",
                               "Read": "Relax",
                               "Watch_TV": "Relax",
                               "Leave_Home": "Leave_Home",
                               "Dining_Rm_Activity": "Eat",
                               "Kitchen_Activity": "Cook",
                               "Bed_to_Toilet": "Bed_to_toilet",
                               "Master_Bathroom": "Bathing",
                               "Guest_Bathroom": "Bathing"},
                     }

datasets = [
    "./datasets/casas/cairo/data",
    "./datasets/casas/twor.2009/data",
    "./datasets/casas/twor.2010/data",
    "./datasets/casas/twor.summer.2009/annotated",
    "./datasets/casas/milan/data"
]
datasetsNames = [i.split('/')[-1] for i in datasets]


def load_raw_data(filename):
    # dateset fields
    timestamps = []
    sensors = []
    values = []
    activities = []

    activity = ''  # empty

    with open(filename, 'rb') as features:
        database = features.readlines()
        for i, line in enumerate(database):  # each line
            f_info = line.decode().split()  # find fields
            try:
                if 'M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][0]:
                    # choose only M D T sensors, avoiding unexpected errors
                    if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                        f_info[1] = f_info[1] + '.000000'
                    timestamps.append(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                        "%Y-%m-%d%H:%M:%S.%f"))
                    sensors.append(str(np.array(f_info[2])))
                    values.append(str(np.array(f_info[3])))

                    if len(f_info) == 4:  # if activity does not exist
                        activities.append(activity)
                    else:  # if activity exists
                        des = str(' '.join(np.array(f_info[4:])))
                        if 'begin' in des:
                            activity = re.sub('begin', '', des)
                            if activity[-1] == ' ':  # if white space at the end
                                activity = activity[:-1]  # delete white space
                            activities.append(activity)
                        if 'end' in des:
                            activities.append(activity)
                            activity = ''
            except IndexError:
                print(i, line)
    features.close()
    # dictionaries: assigning keys to values
    temperature = []
    for element in values:
        try:
            temperature.append(float(element))
        except ValueError:
            pass
    sensorsList = sorted(set(sensors))
    dictSensors = {}
    for i, sensor in enumerate(sensorsList):
        dictSensors[sensor] = i
    activityList = sorted(set(activities))
    dictActivities = {}
    for i, activity in enumerate(activityList):
        dictActivities[activity] = i
    valueList = sorted(set(values))
    dictValues = {}
    for i, v in enumerate(valueList):
        dictValues[v] = i
    dictObs = {}
    count = 0
    for key in dictSensors.keys():
        if "M" or "AD" in key:
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1
        if "D" in key:
            dictObs[key + "CLOSE"] = count
            count += 1
            dictObs[key + "OPEN"] = count
            count += 1
        if "T" in key:
            for temp in range(0, int((max(temperature) - min(temperature)) * 2) + 1):
                dictObs[key + str(float(temp / 2.0) + min(temperature))] = count + temp

    XX = []
    YY = []
    X = []
    Y = []
    for kk, s in enumerate(sensors):
        if "T" in s:
            XX.append(dictObs[s + str(round(float(values[kk]), 1))])
        else:
            XX.append(dictObs[s + str(values[kk])])
        YY.append(dictActivities[activities[kk]])

    x = []
    for i, y in enumerate(YY):
        if i == 0:
            Y.append(y)
            x = [XX[i]]
        if i > 0:
            if y == YY[i - 1]:
                x.append(XX[i])
            else:
                Y.append(y)
                X.append(x)
                x = [XX[i]]
        if i == len(YY) - 1:
            if y != YY[i - 1]:
                Y.append(y)
            X.append(x)
    return X, Y, dictActivities


def load_dataset_v2(filename):
    # Read the file into a DataFrame
    df = pd.read_csv(filename, delimiter='\t', header=None, dtype=str,
                     names=['time', 'sensor', 'value', 'activity'], na_filter=False)

    # Filter the DataFrame to include only M, D, and T sensors
    df = df[df['sensor'].str.startswith(('M', 'D', 'T'))]

    # Ensure that the time format has microseconds
    df['time'] = df['time'].apply(lambda x: x if '.' in x else x + '.000000')

    # Combine date and time and parse as datetime
    df['timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')

    # Identify activities in the DataFrame
    # df['activity'] = np.where(df['sensor'].str.startswith(('M', 'D', 'T')), '', df.iloc[:, 4:].apply(' '.join, axis=1))
    activities = []
    for act in df.activity:
        if len(activities) == 0:
            activities.append('')
        else:
            if 'begin' in act:
                act = act.replace('begin', '').strip()
                activities.append(act)
            elif 'end' in act:
                activities.append('')
            else:
                activities.append(activities[-1])
    df['activity'] = activities

    # Drop unnecessary columns and rows with missing sensor data
    df = df[df['sensor'].str.startswith(('M', 'D', 'T'))].drop(columns=['time'])

    # Create dictionaries for sensors, activities, and values
    dictSensors = {sensor: i for i, sensor in enumerate(sorted(df['sensor'].unique()))}
    dictActivities = {activity: i for i, activity in enumerate(sorted(df['activity'].unique()))}
    dictValues = {value: i for i, value in enumerate(sorted(df['value'].unique()))}

    # Assign encoded values to the DataFrame
    df['XX'] = df.apply(
        lambda row: dictValues[row['value']] if 'T' in row['sensor'] else dictValues[row['value']] + dictSensors[
            row['sensor']], axis=1)
    df['YY'] = df['activity'].apply(lambda x: dictActivities[x])

    # Group the DataFrame by activity and create lists of XX and YY values
    grouped = df.groupby('activity')
    X = grouped['XX'].apply(list).values.tolist()
    Y = grouped['YY'].first().values.tolist()

    return X, Y, dictActivities


def convertActivities(X, Y, dictActivities, uniActivities, cookActivities):
    Yf = Y.copy()
    Xf = X.copy()
    activities = {}
    for i, y in enumerate(Y):
        convertact = [key for key, value in dictActivities.items() if value == y][0]
        activity = uniActivities[convertact]
        Yf[i] = int(cookActivities[activity] - offset)
        activities[activity] = Yf[i]

    return Xf, Yf, activities


def process():
    X_d = []
    Y_d = []
    cl_idx = {}
    for i, filename in enumerate(datasets):
        datasetName = filename.split("/")[-2]
        print('Loading ' + datasetName + ' dataset ...')
        X, Y, dictActivities = load_raw_data(filename)

        X, Y, dictActivities = convertActivities(X, Y,
                                                 dictActivities,
                                                 mappingActivities[datasetName],
                                                 cookActivities[datasetName])
        print(dictActivities)
        print("n° instances post-filtering:\t" + str(len(X)))

        print(Counter(Y))

        X = np.array(X, dtype=object)
        Y = np.array(Y, dtype=object)

        X = pad_sequences(X, maxlen=max_lenght, dtype='int32')
        if not os.path.exists('./datasets/casas/npy'):
            os.makedirs('./datasets/casas/npy')

        X_d.append(X)
        rev_activities_dict = {v: k for k, v in dictActivities.items()}
        Y_d += [rev_activities_dict[y] for y in Y]

        print(X.shape, Y.shape)

        np.save('./datasets/casas/npy/' + datasetName + '-x.npy', X)
        np.save('./datasets/casas/npy/' + datasetName + '-y.npy', Y)
        np.save('./datasets/casas/npy/' + datasetName + '-labels.npy', dictActivities)
    j = 0
    for i, v in enumerate(X_d):
        cl_idx[i] = list(range(j, j + len(v)))
        j += len(v)
    X = np.concatenate(X_d)
    dict_activities = {v: k for k, v in enumerate(set(Y_d))}
    Y = np.array([dict_activities[y] for y in Y_d])
    print(dict_activities)
    print(X.shape, Y.shape)
    np.save('./datasets/casas/npy/' + 'all' + '-x.npy', X)
    np.save('./datasets/casas/npy/' + 'all' + '-y.npy', Y)
    np.save('./datasets/casas/npy/' + 'all' + '-labels.npy', dictActivities)
    np.save('./datasets/casas/npy/' + 'all' + '-clidx.npy', cl_idx)


# Main function to download and extract the WidarData.zip file
def main():
    file_path = download_file_from_google_drive(FILE_ID, SAVE_DIR)
    extract_file(file_path, SAVE_DIR)
    process()


if __name__ == "__main__":
    main()
