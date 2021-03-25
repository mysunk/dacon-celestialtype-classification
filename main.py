import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

# %% load dataset
train = pd.read_csv('data_raw/train.csv', index_col=0)
sample_submission = pd.read_csv('data_raw/sample_submission.csv', index_col=0)
test = pd.read_csv('data_raw/test.csv')

# %% make label
column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
def to_number(x, dic):
    return dic[x]
train['type_num'] = train['type'].apply(lambda x: to_number(x, column_number))
train_label = train['type_num']
train = train.drop(columns=['type', 'type_num'], axis=1)
test = test.drop(columns=['id'],axis=1)

# k-fold
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

# sort parameter in ascending order
cv_result = pd.read_csv('gbm_trials_2.csv')
cv_result = cv_result.sort_values('loss')

# for top 10 models
# parameter setting
n_models = 30
y_pred = []
cv_loss = []
resub_loss = []
for i in range(n_models):
    param = {}
    for j in range(cv_result.shape[1]):
        param[cv_result.columns[j]] = cv_result.iloc[i][j]
    del param['loss']
    del param['status']
    param['seed'] = 0

    j=0
    # train-val split
    result_test = []
    result_train = []
    valid_loss = []
    for train_index, test_index in skf.split(train.values, train_label.values):
        lgb_train = lgb.Dataset(train.iloc[train_index,:], label = train_label[train_index], free_raw_data = False)
        lgb_val = lgb.Dataset(train.iloc[test_index,:], label = train_label[test_index], free_raw_data = False)
        gbm = lgb.train(param, lgb_train, num_boost_round=1000,valid_sets = lgb_val, early_stopping_rounds=5)
        result_test.append(gbm.predict(test))
        result_train.append(gbm.predict(train))
        print('============Finished ', i, ' th model,',j,'th validation=====================')
        j=j+1
        valid_loss.append(gbm.best_score['valid_0']['multi_logloss'])
    print('============Finished ',i,' th model train=====================')
    y_pred.append(np.mean(result_test,axis=0))
    cv_loss.append(np.mean(valid_loss,axis=0))
    resub_loss.append(np.mean(result_train,axis=0))

y_pred_mean = np.mean(y_pred,axis=0)
submit = pd.DataFrame(data = y_pred_mean, columns = sample_submission.columns, index = sample_submission.index)
submit.to_csv('submit/submit_6.csv',index=True)