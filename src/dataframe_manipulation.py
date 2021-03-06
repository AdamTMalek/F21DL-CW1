import pandas as pd
import sklearn
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
# To plot pretty figures


# Loads the image and concatenates with the y-data holding class data from 0-9
def load_image_data(x_path, y_path):
    x_data = pd.read_csv(x_path)
    y_data = pd.read_csv(y_path)
    concatenated = pd.concat([x_data, y_data], axis=1)
    return concatenated

# Only load images file and returns it
def load_image_data_without_y_data():
    x_data = pd.read_csv("../data/x_train_gr_smpl.csv")
    return x_data

# Only load images file and returns it
def load_image_data_without_y_data():
    x_data = pd.read_csv("../data/x_train_gr_smpl.csv")
    return x_data


# Loads all the boolean y-data files, concatenates and returns all of them
def load_all_y_data():
    data = pd.DataFrame()
    for x in range(0, 10):
        y = pd.read_csv("data/y_train_smpl_"+str(x)+".csv", header=0)
        data = pd.concat([data, y], axis=1)

    data.columns = ["Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
                    "Left Turn", "Right Turn", "Beware Pedestrian Crossing", "Beware Children", "Beware Cycle Route Ahead"]
    return data


# Loads the image and concatenates with each boolean y-data file, 0=true 1=false
def load_image_data_with_ten_one(x_path):

    x_data = pd.read_csv(x_path)
    y_data = load_all_y_data()
    concatenated = pd.concat([x_data, y_data], axis=1)
    return concatenated

# Splits x and y data from a combined images dataframe
def separate_x_and_y(images):

    x_data = images.iloc[:, images.columns != '0']
    y_data = images.iloc[:, -1:]
    return x_data, y_data

def shuffle_dataframe(data_frame):
    return data_frame.sample(frac=1)
