import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

# Constant to display the graph of the information
DISPLAY_GRAPH = False

# TODO Project
# colors = pd.read_excel("colors.xlsx", usecols=2)
allTable = pd.read_excel("colors.xlsx")

if DISPLAY_GRAPH:
    # Set dark background to the graph
    plt.style.use('dark_background')

    isWhite = allTable[allTable["Class"] == 0][["R", "G", "B"]]
    isYellow = allTable[allTable["Class"] == 1][["R", "G", "B"]]
    isOrange = allTable[allTable["Class"] == 2][["R", "G", "B"]]
    isRed = allTable[allTable["Class"] == 3][["R", "G", "B"]]
    isBlue = allTable[allTable["Class"] == 4][["R", "G", "B"]]
    isGreen = allTable[allTable["Class"] == 5][["R", "G", "B"]]

    whiteBoxes = isWhite.values
    yellowBoxes = isYellow.values.tolist()
    orangeBoxes = isOrange.values.tolist()
    redBoxes = isRed.values.tolist()
    blueBoxes = isBlue.values.tolist()

    colors = allTable[["R", "G", "B"]].values.tolist()

    # Transformation
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(colors)

    # Setting up the figures for plotting data
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)

    ax.set_facecolor((0, 0, 0))

    # Plotting the graphs
    ax.scatter(isWhite["R"].tolist(), isWhite["G"].tolist(), isWhite["B"].tolist(), color="purple", marker='4')
    ax.scatter(isYellow["R"].tolist(), isYellow["G"].tolist(), isYellow["B"].tolist(), color="yellow", marker='o')
    ax.scatter(isOrange["R"].tolist(), isOrange["G"].tolist(), isOrange["B"].tolist(), color="orange", marker='D')
    ax.scatter(isRed["R"].tolist(), isRed["G"].tolist(), isRed["B"].tolist(), c="r", marker='X')
    ax.scatter(isBlue["R"].tolist(), isBlue["G"].tolist(), isBlue["B"].tolist(), color="blue", marker='H')
    ax.scatter(isGreen["R"].tolist(), isGreen["G"].tolist(), isGreen["B"].tolist(), color="green", marker='+')

    # Setting the labels and title
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title("Colors")

    # Setting the axis limits
    ax.set_ylim(0, 300)
    ax.set_xlim(0, 300)
    ax.set_zlim(0, 300)

    # Plotting the graphs
    ax2.set_xlabel("z1")
    ax2.set_ylabel("z2")
    ax2.set_title("Colors after dimensionality reduction")
    colorArr = ["white", "yellow", "orange", "red", "blue", "green"]
    for i in range(len(allTable['Class'].values.tolist()) - 1):
        ax2.scatter((X2D[:, 0].tolist())[i], (X2D[:, 1].tolist())[i],
                    color=colorArr[allTable['Class'].values.tolist()[i]])

    plt.show()


# Here is model for
def train_set(data, test_ratio):
    index = int(test_ratio * len(data))

    return data[index:], data[:index]


def display_num_error(predicted, actual):
    num = 0
    for i in range(len(predicted) - 1):
        if predicted[i] != actual[i]:
            num = num + 1
    return num


train_data_set, test_data_set = train_set(allTable, 0.2)

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(train_data_set[["R", "G", "B"]], train_data_set["Class"])
scores = cross_val_score(svm_clf, allTable[["R", "G", "B"]].values, allTable["Class"].values, cv=10, scoring='accuracy')
print("Scores after cross-validation:", scores)

# Tuning the parameters
clf = SVC()
param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(allTable[["R", "G", "B"]].values, allTable["Class"].values)
res = grid.cv_results_
print(res)
