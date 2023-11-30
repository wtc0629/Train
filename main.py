import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tester_name = ["PHD", "DRP", "QXR", "TMZ", "TYT", "WTC", "WTC2", "ZC", "YKD", "WTC3", "Philipp", "Dennis","1","2","3","Felix"]
#tester_name = ["PHD", "DRP", "QXR", "ZC", "WTC3", "Dennis", "1", "2", "3"]
# tester_name = ["PHD", "QXR", "Philipp", "Dennis", "1", "3"]
#tester_name = ["PHD", "QXR", "Philipp"]


def loo_cross_validation(data_frames, new_y, y):
    loo = LeaveOneOut()

    accuracies = []
    j = 0
    for train_index, test_index in loo.split(data_frames):
        train_data = pd.concat([data_frames[i] for i in train_index])
        test_data = pd.concat([data_frames[i] for i in test_index])

        # 从目标标签中选择对应的数据
        y_train = new_y
        y_test = y

        # 训练和测试模型
        classifier = RandomForestClassifier(max_features=100, random_state=5)
        classifier.fit(train_data.values, y_train.squeeze())
        predictions = classifier.predict(test_data)
        # print(predictions)
        # print(len(predictions))
        # print("1")
        cm = confusion_matrix(y_test, predictions)
        # sns.heatmap(cm,
        #            annot=True,
        #            fmt='g')
        # plt.ylabel('Prediction', fontsize=13)
        # plt.xlabel('Actual', fontsize=13)
        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
        #    "hor", "lrc", "lc", "ntf", "rc", "sq", "ver"])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                     "fhor", "hor", "lrc", "lc", "fntf", "ntf", "rc", "fsq", "sq", "fver", "ver"])
        cm_display.plot()
        plt.title('Confusion Matrix', fontsize=17)
        plt.savefig(f"C:\\Users\\51004\\Desktop\\MergeCSV\\{tester_name[j]}\\ConfusionMatrix.png")
        plt.show()
        j = j + 1

        # 计算准确率
        accuracy = accuracy_score(y_test.squeeze(), predictions)
        accuracies.append(accuracy)

    return accuracies


if __name__ == "__main__":

    data_frames = []
    # for i in range(15):
    # for i in range(6):
    # for i in range(3):
    y_index = []
    for i in range(len(tester_name)):
        filename = f"C:\\Users\\51004\\Desktop\\MergeCSV\\{tester_name[i]}\\train_data.csv"
        data = pd.read_csv(filename)
        features = extract_features(data, column_id="process", column_sort="gaze_timestamp_datetime")
        features = impute(features)
        y_index = list(features.index)
        data_frames.append(features)



    gestures2 = ["horizontally fast", "horizontally",
                 "large right circle (clockwise)", "left circle (anticlockwise)", "near to far fast",
                 "near to far", "right circle (clockwise)", "square fast",
                 "square", "vertically fast", "vertically"]
    # gestures2 = ["horizontally", "horizontally",
    #             "large right circle (clockwise)", "left circle (anticlockwise)", "near to far",
    #            "near to far", "right circle (clockwise)", "square",
    #             "square", "vertically", "vertically"]
    y_gestures = []
    for i in range(11):
        for j in range(32):
            y_gestures.append(gestures2[i])
    y_data = pd.DataFrame({'gesture': y_gestures},
                          index=y_index)
    y = y_data.squeeze()

    # print(y)
    # print("0")
    new_y_dataframes = []
    # for i in range(14):
    # for i in range(5):
    # for i in range(2):
    for i in range(len(tester_name) - 1):
        new_y_dataframes.append(y_data)
    new_y_data = pd.concat(new_y_dataframes)
    new_y = new_y_data.squeeze()

    accuracies = loo_cross_validation(data_frames, new_y, y)

    plt.close('all')
    print(accuracies)
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Accuracy: {average_accuracy}")
