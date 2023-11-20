import pandas as pd
from sklearn.model_selection import KFold
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

gestures = ["horizontally", "horizontally fast", "vertically", "vertically fast", "near to far", "near to far fast",
            "square", "square fast", "left circle (anticlockwise)", "right circle (clockwise)",
            "large right circle (clockwise)"]


if __name__ == "__main__":
    y_index = ['horizontally fast_GestureTime0', 'horizontally fast_GestureTime1',
               'horizontally fast_GestureTime2', 'horizontally fast_GestureTime3',
               'horizontally fast_GestureTime4', 'horizontally fast_GestureTime5',
               'horizontally fast_GestureTime6', 'horizontally fast_GestureTime7',
               'horizontally_GestureTime0', 'horizontally_GestureTime1',
               'horizontally_GestureTime2', 'horizontally_GestureTime3',
               'horizontally_GestureTime4', 'horizontally_GestureTime5',
               'horizontally_GestureTime6', 'horizontally_GestureTime7',
               'large right circle (clockwise)_GestureTime0',
               'large right circle (clockwise)_GestureTime1',
               'large right circle (clockwise)_GestureTime2',
               'large right circle (clockwise)_GestureTime3',
               'large right circle (clockwise)_GestureTime4',
               'large right circle (clockwise)_GestureTime5',
               'large right circle (clockwise)_GestureTime6',
               'large right circle (clockwise)_GestureTime7',
               'left circle (anticlockwise)_GestureTime0',
               'left circle (anticlockwise)_GestureTime1',
               'left circle (anticlockwise)_GestureTime2',
               'left circle (anticlockwise)_GestureTime3',
               'left circle (anticlockwise)_GestureTime4',
               'left circle (anticlockwise)_GestureTime5',
               'left circle (anticlockwise)_GestureTime6',
               'left circle (anticlockwise)_GestureTime7',
               'near to far fast_GestureTime0', 'near to far fast_GestureTime1',
               'near to far fast_GestureTime2', 'near to far fast_GestureTime3',
               'near to far fast_GestureTime4', 'near to far fast_GestureTime5',
               'near to far fast_GestureTime6', 'near to far fast_GestureTime7',
               'near to far_GestureTime0', 'near to far_GestureTime1',
               'near to far_GestureTime2', 'near to far_GestureTime3',
               'near to far_GestureTime4', 'near to far_GestureTime5',
               'near to far_GestureTime6', 'near to far_GestureTime7',
               'right circle (clockwise)_GestureTime0',
               'right circle (clockwise)_GestureTime1',
               'right circle (clockwise)_GestureTime2',
               'right circle (clockwise)_GestureTime3',
               'right circle (clockwise)_GestureTime4',
               'right circle (clockwise)_GestureTime5',
               'right circle (clockwise)_GestureTime6',
               'right circle (clockwise)_GestureTime7', 'square fast_GestureTime0',
               'square fast_GestureTime1', 'square fast_GestureTime2',
               'square fast_GestureTime3', 'square fast_GestureTime4',
               'square fast_GestureTime5', 'square fast_GestureTime6',
               'square fast_GestureTime7', 'square_GestureTime0',
               'square_GestureTime1', 'square_GestureTime2', 'square_GestureTime3',
               'square_GestureTime4', 'square_GestureTime5', 'square_GestureTime6',
               'square_GestureTime7', 'vertically fast_GestureTime0',
               'vertically fast_GestureTime1', 'vertically fast_GestureTime2',
               'vertically fast_GestureTime3', 'vertically fast_GestureTime4',
               'vertically fast_GestureTime5', 'vertically fast_GestureTime6',
               'vertically fast_GestureTime7', 'vertically_GestureTime0',
               'vertically_GestureTime1', 'vertically_GestureTime2',
               'vertically_GestureTime3', 'vertically_GestureTime4',
               'vertically_GestureTime5', 'vertically_GestureTime6',
               'vertically_GestureTime7']

    gestures2 = ["horizontally fast", "horizontally",
                 "large right circle (clockwise)", "left circle (anticlockwise)", "near to far fast",
                 "near to far", "right circle (clockwise)", "square fast",
                 "square", "vertically fast", "vertically"]
    y_gestures = []
    for i in range(11):
        for j in range(8):
            y_gestures.append(gestures2[i])
    y_data = pd.DataFrame({'gesture': y_gestures},
                          index=y_index)
    y = y_data.squeeze()
    new_y_dataframes = []
    for i in range(11):
        new_y_dataframes.append(y_data)
    new_y_data = pd.concat(new_y_dataframes)
    new_y = new_y_data.squeeze()

    tester_name = ["PHD", "DRP", "QXR", "TMZ", "TYT", "WTC", "WTC2", "ZC", "YKD", "WTC3", "Philipp", "Dennis"]
    data_frames = []

    for i in range(11):
        filename = "C:\\Users\\51004\\Desktop\\MergeCSV\\" + tester_name[i] + "\\train_data.csv"
        data = pd.read_csv(filename)
        features = extract_features(data, column_id="process", column_sort="gaze_timestamp_datetime")
        features = impute(features)
        #features_filtered = select_features(features, y)
        #features_filtered = impute(features_filtered)
        data_frames.append(features)
        #data_frames.append(features)

    # 合并11个数据帧为一个训练数据帧
    train_data = pd.concat(data_frames)
    #train_data.fillna(0, inplace=True)
    print(train_data)
    train_data.to_csv("C:\\Users\\51004\\Desktop\\MergeCSV\\Dennis\\ex1.csv")
    # 准备目标标签
    # y_gestures = ["gesture_1", "gesture_2", ..., "gesture_11"]
    # y_data = pd.DataFrame({'gesture': y_gestures})

    # 训练随机森林分类器
    classifier = RandomForestClassifier()
    classifier.fit(train_data.values, new_y)

    # 读取第12个gaze_merged.csv文件并提取特征值
    test_data = pd.read_csv("C:\\Users\\51004\\Desktop\\MergeCSV\\" + tester_name[11] + "\\train_data.csv")
    test_features = extract_features(test_data, column_id="process", column_sort="gaze_timestamp_datetime")
    test_features = impute(test_features)
    #test_features_filtered = select_features(test_features, y)
    #test_features_filtered = impute(test_features_filtered)
    #test_features_filtered.fillna(0, inplace=True)
    #test_features.fillna(0, inplace=True)
    # 使用训练好的分类器进行预测
    predictions = classifier.predict(test_features)

    # 验证准确率
    # true_labels = ["true_gesture_1", "true_gesture_2", ..., "true_gesture_n"]  # 第12个文件的真实标签
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")
