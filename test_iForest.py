import pandas as pd
from sklearn.ensemble import IsolationForest

ilf = IsolationForest(n_estimators=100,
                      n_jobs=1,  # 使用全部cpu
                      verbose=2,
                      )
label = pd.read_csv('D:\\test_data\\guiji\\text_cnn_label2.csv')
data = pd.read_csv('D:\\test_data\\guiji\\text_cnn2.csv')
data = data.fillna(0)
# 选取特征，不使用标签(类型)
print(data.head())
X_cols = ['x_time_down0','y_time_down0','count_x_t1subt2','count_y_t1subt2','abs_k','time_all']

# 训练
ilf.fit(data[X_cols])
shape = data.shape[0]
batch = 10 ** 5

all_pred = []
for i in range(int(shape / batch)):
    start = i * batch
    end = (i + 1) * batch
    test = data[X_cols][start:end]
    # 预测
    pred = ilf.predict(test)
    all_pred.extend(pred)

data['pred'] = all_pred
# data.to_csv('E:\\datas\\guiji_for_text_cnn\\iforest_pre.csv', columns=["pred", ], header=False)
print(data.head())