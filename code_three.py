# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from scikeras.wrappers import KerasRegressor
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset
from pytorch_tabnet.tab_model import TabNetRegressor
import tensorflow as tf
from pytorch_tabnet.metrics import Metric
from catboost import CatBoostRegressor
from tensorflow.keras.regularizers import l2, l1

def seed(my_seed):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.benchmark = False  # This can slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
seed(42)

def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)
    
class CustomMetric(Metric):
    def __init__(self):
        self._name = "r2"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        return r2

def calMetrics(real, pred):
    rmse = np.sqrt(mean_squared_error(real, pred))
    r2 = r2_score(real, pred)
    mae = np.mean(np.abs(real - pred))
    percentage_errors = ((real - pred) / real) * 100
    mpe = np.mean(percentage_errors)
    return rmse, r2, mae, mpe

class Transformer(torch.nn.Module):
    def __init__(self,input_dim, hidden_nums, drop_out):
        super(Transformer, self).__init__()  #
        # hidden_nums = 100
        self.fc1 = torch.nn.Linear(input_dim, hidden_nums)
        # self.nl_1 = NONLocalBlock1D(in_channels=1)
        self.fc2 = torch.nn.Linear(hidden_nums, hidden_nums)
        # self.nl_2 = NONLocalBlock1D(in_channels=1)
        self.fc3 = torch.nn.Linear(hidden_nums, hidden_nums)
        # self.nl_3 = NONLocalBlock1D(in_channels=1)
        self.fc4= torch.nn.Linear(hidden_nums, hidden_nums)
        # self.nl_4 = NONLocalBlock1D(in_channels=1)
        self.fc5 = torch.nn.Linear(hidden_nums, hidden_nums)
        # self.nl_5 = NONLocalBlock1D(in_channels=1)
        # self.fc4 = torch.nn.Linear(1000, 1000)
        self.fc = torch.nn.Linear(hidden_nums, 1)
        self.BN1 = nn.BatchNorm1d(hidden_nums)
        # self.BN2 = nn.BatchNorm1d(300)
        # self.BN3 = nn.BatchNorm1d(300)
        # self.BN4 = nn.BatchNorm1d(300)
        # self.BN5 = nn.BatchNorm1d(300)
        self.dr = nn.Dropout(drop_out)
        # self.ac = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, din):
        feature_1 = self.dr(self.BN1(self.fc1(din)))
        # print(feature_1)
        
        # feature_1 = self.fc1(din)
        # feature_1 = self.fc2(feature_1)
        # feature_1 = self.fc3(feature_1)
        # feature_1 = self.dr(self.BN1(feature_1))
        
        return self.fc(feature_1), 1

def trainModel(model, optimizer, epochs, trainloader):
    for epoch in range(epochs):
        model.train()  ## Model is in train mode (look at pytorch library to see meaning)
        total_loss = 0
        all_preds = []
        all_labels = []
        for i, (batch, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            src = batch.type(torch.float32)  # Turn into a batch
            preds, nl_map_0 = model(src)   # src.cuda()
            preds_after = preds[:].squeeze()
            
            preds_after_reshape = preds_after.view(-1, 1)
            labels_reshape = labels.view(-1, 1)
            loss = nn.MSELoss()(preds_after_reshape, labels_reshape)  # labels.cuda()
            all_preds.append(preds_after)
            all_labels.append(labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
        print(str(epoch) + ': ' +'-----------------RMSE: ' + str(np.sqrt(loss.data)))
    # return model

def testModel(model, testloader):
    total_loss = 0
    model.eval()
    all_preds = []
    all_labels = []
    all_data = []
    with torch.no_grad():
        for i, (batch, labels) in enumerate(testloader):
            src = batch.type(torch.float32)  # Turn into a batch
            # time1 = time.time()
            preds, nl_map_0 = model(src)
            preds_after = preds[:].squeeze()
            # preds_soft = F.softmax(preds, dim=-1)
            # time2 = time.time()
            preds_after_reshape = preds_after.view(-1, 1)
            labels_reshape = labels.view(-1, 1)
            loss = nn.MSELoss()(preds_after_reshape, labels_reshape)
            all_preds.append(preds_after)
            all_labels.append(labels)
            all_data.append(src)
            total_loss += loss.data
    real_results = torch.cat(all_labels, dim=0).flatten().type(torch.float32).cpu().numpy()
    pred_results = torch.cat(all_preds, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    # data_results = torch.cat(all_data, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    print(real_results.shape, pred_results.shape)
    np.sqrt(mean_squared_error(real_results, pred_results))
    r2_score(real_results, pred_results)
    np.mean(np.abs(real_results - pred_results))

    rmse, r2, mae, mpe = calMetrics(real_results, pred_results)
    return rmse, r2, mae, mpe, pred_results, real_results

def getWideDeepModel(linear_inputs, dnn_inputs, y_train, linear_epochs=400, dnn_epochs=500, combined_epochs=800):
    linear_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(linear_inputs.shape[1],)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    # linear_model.compile('adagrad', 'mse')
    linear_model.compile(optimizer='adam', loss='mean_squared_error')
    linear_model.fit(linear_inputs, y_train, epochs=linear_epochs, batch_size=10)
    # dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    dnn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='linear', kernel_regularizer=l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=32, activation='linear'),
        tf.keras.layers.Dense(units=1)
    ])
    dnn_model.compile('rmsprop', 'mse')
    dnn_model.fit(dnn_inputs, y_train, epochs=dnn_epochs, batch_size=64)
    combined_model = tf.keras.experimental.WideDeepModel(linear_model, dnn_model)
    combined_model.compile(optimizer=['sgd', 'adam'], loss='mse', metrics=['mse'])
    combined_model.fit([linear_inputs, dnn_inputs], y_train, epochs=combined_epochs, batch_size=64)
    return combined_model

# %% read data
df = pd.read_excel(r'df_three.xlsx')
cols = df.columns.values

# %% split data
X = df.drop('AUC', axis=1)
y = df['AUC']
# 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# %% model
#### GBM
other_params = other_params = {'n_estimators': 100,
 'learning_rate': 0.09,
 'max_depth': 3,
 'random_state': 42}
best_gbm = GradientBoostingRegressor(**other_params)
best_gbm.fit(X_train, y_train)
y_pred_gbm = best_gbm.predict(X_test)

#### RF
other_params = {'n_estimators': 500,
 'max_depth': 9,
 'min_samples_split': 3,
 'min_samples_leaf': 3,
 'max_features': 10,
 'random_state': 42}
best_rf = RandomForestRegressor(**other_params)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

#### XGBoost
other_params = {'learning_rate': 0.1,
 'n_estimators': 100,
 'max_depth': 4,
 'min_child_weight': 1,
 'seed': 0,
 'subsample': 0.5,
 'colsample_bytree': 0.8,
 'gamma': 0.6,
 'reg_alpha': 0.1,
 'reg_lambda': 1,
 'random_state': 42}
best_xgb = XGBRegressor(**other_params)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

#### LightGBM
other_params = {'num_leaves': 10,
 'n_estimators': 100,
 'max_depth': 5,
 'learning_rate': 0.1,
 'max_bin': 100,
 'min_data_in_leaf': 1,
 'random_state': 42}
best_lgb = lgb.LGBMRegressor(**other_params)
best_lgb.fit(X_train, y_train)
y_pred_lgb = best_lgb.predict(X_test)

#### GBDT
other_params = {'n_estimators': 100,
 'max_depth': 5,
 'subsample': 0.8,
 'max_features': 15,
 'learning_rate': 0.1,
 'random_state': 42}
best_gbdt = GradientBoostingRegressor(**other_params)
best_gbdt.fit(X_train, y_train)
y_pred_gbdt = best_gbdt.predict(X_test)

#### CatBoost
best_params_cbt = {'depth': 3, 'iterations': 300, 'learning_rate': 0.1, 'random_state': 30}
best_cbt = CatBoostRegressor(**best_params_cbt)
best_cbt.fit(X_train, y_train)
y_pred_cbt = best_cbt.predict(X_test)

#### ANN
def get_reg(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(Dense(hidden_layer_size, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    return model
best_ann = KerasRegressor(
    model=get_reg,
    loss="mse",  # mse
    metrics=[KerasRegressor.r_squared],
    hidden_layer_sizes=(50, 50),
    dropout=0,
    epochs=150
)
best_ann.fit(X_train, y_train)
y_pred_ann = best_ann.predict(X_test)

#### Transformer
train_dataset = TensorDataset(torch.Tensor(X_train.to_numpy().astype(float)),
                              torch.Tensor(y_train.values))

test_dataset = TensorDataset(torch.Tensor(X_test.to_numpy().astype(float)),
                              torch.Tensor(y_test.values))
batch_size = 64
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=False
                                          , num_workers=0, worker_init_fn=_init_fn)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False
                                          , num_workers=0, worker_init_fn=_init_fn)
input_dim = X.shape[1]

all_dataset = TensorDataset(torch.Tensor(X.to_numpy().astype(float)),
                              torch.Tensor(y.values))
all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size,
                                          shuffle=False
                                          , num_workers=0, worker_init_fn=_init_fn)

best_hidden_nums, best_drop_out, best_epochs = 200, 0.1, 200
best_transformer = Transformer(input_dim = input_dim, hidden_nums=best_hidden_nums, drop_out=best_drop_out)
optimizer = torch.optim.Adam(best_transformer.parameters(), lr=0.001)
trainModel(best_transformer, optimizer, best_epochs, trainloader)
rmse_tf, r2_tf, mae_tf, mpe_tf, pred_results_tf, real_results_tf = testModel(best_transformer, testloader)

#### Tabnet
cat_idxs = [0,4,5,6]
cat_dims = [7,2,2,2]
best_nd, best_na = 10, 10
best_n_steps = 5
best_tabnet = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, seed=42,\
                        n_d=best_nd, n_a=best_na, n_steps=best_n_steps)
max_epochs = 100
batch_size = 32
best_tabnet.fit(
    X_train=X_train.values,
    y_train=y_train.values.reshape(-1, 1),
    eval_set=[(X_test.values, y_test.values.reshape(-1, 1))],
    eval_name=['valid'],
    eval_metric=['rmse', CustomMetric],
    max_epochs=max_epochs,
    patience=50, batch_size=batch_size
)
y_pred_tb = best_tabnet.predict(X_test.values)

#### Wide&Deep
catCols = ['daily_dose', 'C4h>C1.5h', 'C9h>C1.5h', 'C9h>C4h']
conCols = ['C1.5h', 'C4h', 'C9h', 'ALT', 'AST', 'cr']
X_train_cat = X_train[catCols]
X_test_cat = X_test[catCols]
X_train_con = X_train[conCols]
X_test_con = X_test[conCols]
best_com_epochs = 600
linear_inputs = X_train_cat
dnn_inputs = X_train_con
best_wd = getWideDeepModel(linear_inputs, dnn_inputs, y_train, combined_epochs=best_com_epochs)

y_pred_wd = best_wd.predict([X_test_cat, X_test_con])


# %% y-pred
yPredDF = pd.DataFrame(columns=['GBM', 'RF', 'XGBoost', 'LightGBM', 'GBDT', 'CatBoost', 'ANN', 'TabNet', 'Wide&Deep', 'y_test'])
yPredDF['GBM'] = y_pred_gbm
yPredDF['RF'] = y_pred_rf
yPredDF['XGBoost'] = y_pred_xgb
yPredDF['LightGBM'] = y_pred_lgb
yPredDF['GBDT'] = y_pred_gbdt
yPredDF['CatBoost'] = y_pred_cbt
yPredDF['ANN'] = y_pred_ann
yPredDF['TabNet'] = y_pred_tb
yPredDF['Wide&Deep'] = y_pred_wd
y_test_1 = y_test.reset_index(drop=True)
yPredDF['y_test'] = y_test_1

yPredDF['Transformer'] = pred_results_tf
yPredDF['Transformer_true'] = real_results_tf

# %% metric
metricDF = pd.DataFrame(columns=['Algorithm', 'RMSE', 'R2', 'MAE', 'MPE'])
algL = ['GBM', 'RF', 'XGBoost', 'LightGBM', 'GBDT', 'CatBoost', 'ANN', 'Transformer', 'TabNet', 'Wide&Deep']
for alg in algL:
    true = yPredDF['y_test']
    pred = yPredDF[alg]
    if alg == 'Transformer':
        true = yPredDF['Transformer_true']
    rmse, r2, mae, mpe = calMetrics(true, pred)
    metricDF = metricDF.append({'Algorithm': alg, 'RMSE': rmse, 'R2': r2, 'MAE': mae, \
                                'MPE': mpe}, ignore_index=True)








