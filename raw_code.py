# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:25:47 2017

@author: gleam
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import precision_score,recall_score

##读取原始数据并预处理
def get_data():
    action02=pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_Action_201602.csv",encoding="GBK") 
    action03=pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_Action_201603.csv",encoding="GBK")
    action04=pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_Action_201604.csv",encoding="GBK")
    return pd.concat([action02,action03,action04],axis=0)
    
action = get_data()
action['user_id']=action['user_id'].astype(np.str)
action['user_id']=action['user_id'].apply(lambda x:x[:len(x)-2])
action=action.reset_index(drop=True)
action['date']=pd.to_datetime(action['time'].str[:10])
action['time']=pd.to_datetime(action['time'])

##数据集划分
def data_filter(df,start_date,end_date):
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    return df.ix[(df['date']>=start_date)&(df['date']<=end_date),:]

produc = pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_Product.csv",encoding = "GBK")
comment = pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_Comment.csv",encoding = "GBK")

##提取特征
User=pd.read_csv(r"E:\maching learning\JD\JD\data\data_raw\JData_User.csv",encoding="GBK")
User_arr=np.array(User['user_id'].unique())
User['user_reg_tm']=pd.to_datetime(User['user_reg_tm'])
def get_feature(last_date):
    User_s=User.copy()
    last_date=pd.to_datetime(last_date)
    User_s.index=User['user_id']
    User_s['f1']=User['user_lv_cd']#用户数据集-用户等级
    User_s['f2']=User['age']#用户数据集-用户年龄
    User_s['f3']=User['sex']#用户数据集-用户性别
    #User['f4']=User['user_reg_tm'].apply(lambda x :(pd.to_datetime('2016-02-01')-x).days)
    features={}
    for j in range(5,38):
        features['f'+str(j)]=[]
    count=0
    for i in User_arr:#提前将每个用户的行为放在一个单独的excel中，对每个excel提特征
        count+=1
        if count%1000==0:
            print(count)
        user_id=str(i)
        if os.path.exists('C:/Users/gleam/Desktop/user/'+user_id+'.csv'):
            df=pd.read_csv('C:/Users/gleam/Desktop/user/'+user_id+'.csv',encoding="GBK")
            df['date']=pd.to_datetime(df['date'])
            df=df.ix[df['date']<=last_date,:]
            ##feartures
            if len(df)>0:
                tem=df.ix[df['type']==4,:]
                if len(tem)==0:#判断该用户是否有过购买记录
                    type4_days=999
                else:
                    type4_days=(last_date-tem.iloc[-1]['date']).days
                features['f5'].append(type4_days)#最近一次下单距今天数
                features['f6'].append((last_date-df.iloc[-1]['date']).days)#最后一次行为距今天数
                features['f7'].append(len(df.ix[df['type']==4,:]))#总下单次数
                features['f8'].append(len(df))#总行为次数
                features['f9'].append(len(df.ix[df['date']>(last_date-pd.Timedelta('1 days 00:00:00')),:]))#最近一天总行为次数
                features['f10'].append(len(df.ix[(df['date']>(last_date-pd.Timedelta('1 days 00:00:00')))&(df['cate']==8),:]))#最近一天cate8行为次数
                features['f11'].append(len(df.ix[df['date']>(last_date-pd.Timedelta('3 days 00:00:00')),:]))#最近三天行为次数
                features['f12'].append(len(df.ix[(df['date']>(last_date-pd.Timedelta('3 days 00:00:00')))&(df['cate']==8),:]))#最近三天cate8行为次数
                features['f13'].append(len(df.ix[df['date']>(last_date-pd.Timedelta('7 days 00:00:00')),:]))#最近七天行为次数
                features['f14'].append(len(df.ix[(df['date']>(last_date-pd.Timedelta('7 days 00:00:00')))&(df['cate']==8),:]))#最近七天cate8行为次数
                features['f15'].append(len(df.ix[df['date']>(last_date-pd.Timedelta('30 days 00:00:00')),:]))#最近30天行为次数
                features['f16'].append(len(df.ix[(df['date']>(last_date-pd.Timedelta('30 days 00:00:00')))&(df['cate']==8),:]))#最近30天cate8行为次数
                features['f17'].append(len(df.ix[df['date']>(last_date-pd.Timedelta('7 days 00:00:00')),:]['date'].unique()))#最近7天中有行为的天数
                features['f18'].append(len(df.ix[(df['date']>(last_date-pd.Timedelta('7 days 00:00:00')))&(df['cate']==8),:]['date'].unique()))#最近7天中对cate8有行为的天数
                tem=df.ix[df['date']>(last_date-pd.Timedelta('3 days 00:00:00')),:]#最近三天的行为数据

                if len(tem)==0:#判断最近三天有无行为
                    sku_rate_3days=0
                    sku_rate_3days_cate8=0
                else:
                    tem1=tem.groupby(['sku_id']).count()
                    tem1=tem1.sort_values(by=['type'],ascending=False)
                    sku_rate_3days=tem1.iloc[0]['type']/len(tem)#最近三天行为最多商品行为占比
                    tem2=tem.ix[tem['cate']==8,:]
                    if len(tem2)==0:
                        sku_rate_3days_cate8=0
                    else:
                        tem2=tem2.groupby(['sku_id']).count()
                        tem2=tem2.sort_values(by=['type'],ascending=False)
                        sku_rate_3days_cate8=tem2.iloc[0]['type']/len(tem)#最近三天行为最多cate8商品行为占比                
                features['f19'].append(sku_rate_3days)
                features['f20'].append(sku_rate_3days_cate8)
                
                tem=df.ix[(df['date']>(last_date-pd.Timedelta('3 days 00:00:00')))&(df['cate']==8),:]
                if len(tem)==0:
                    sku_rate_3days_cate8_s=0
                else:
                    tem1=tem.groupby(['sku_id']).count()
                    tem1=tem1.sort_values(by=['type'],ascending=False)
                    sku_rate_3days_cate8_s=tem1.iloc[0]['type']/len(tem)#最近三天关注最多cate8商品占cate8行为的比
                features['f21'].append(sku_rate_3days_cate8_s)
                
                tem=df.ix[df['date']>(last_date-pd.Timedelta('1 days 00:00:00')),:]
                if len(tem)==0:
                    sku_rate_1days=0
                    sku_rate_1days_cate8=0
                else:
                    tem1=tem.groupby(['sku_id']).count()
                    tem1=tem1.sort_values(by=['type'],ascending=False)
                    sku_rate_1days=tem1.iloc[0]['type']/len(tem)#最近一天行为最多商品行为占比
                    tem2=tem.ix[tem['cate']==8,:]
                    if len(tem2)==0:
                        sku_rate_1days_cate8=0
                    else:
                        tem2=tem2.groupby(['sku_id']).count()
                        tem2=tem2.sort_values(by=['type'],ascending=False)
                        sku_rate_1days_cate8=tem2.iloc[0]['type']/len(tem)#最近一天行为最多cate8商品行为占比                
                features['f22'].append(sku_rate_1days)
                features['f23'].append(sku_rate_1days_cate8)
                
                tem=df.ix[(df['date']>(last_date-pd.Timedelta('1 days 00:00:00')))&(df['cate']==8),:]
                if len(tem)==0:
                    sku_rate_1days_cate8_s=0
                else:
                    tem1=tem.groupby(['sku_id']).count()
                    tem1=tem1.sort_values(by=['type'],ascending=False)
                    sku_rate_1days_cate8_s=tem1.iloc[0]['type']/len(tem)#最近一天关注最多cate8商品占cate8行为的比
                features['f24'].append(sku_rate_1days_cate8_s)            
                
                tem=df.ix[df['cate']==8,:]
                features['f25'].append(len(tem)/len(df))#历史cate8行为数占比 
                features['f26'].append(len(df['sku_id'].unique()))#历史有行为商品个数
                features['f27'].append(len(tem['sku_id'].unique()))#历史有行为cate8商品个数
                
                tem=df.groupby(['sku_id']).count()
                tem=tem.sort_values(by=['type'],ascending=False)
                features['f28'].append(tem.iloc[0]['type'])#历史行为最多商品的行为数量
                
                tem=df.ix[df['cate']==8,:]
                if len(tem)==0:
                    features['f29'].append(0)
                else:
                    tem=tem.groupby(['sku_id']).count()
                    tem=tem.sort_values(by=['type'],ascending=False)
                    features['f29'].append(tem.iloc[0]['type'])#历史行为最多cate8商品的行为数量
                    
                tem=df.ix[df['type']==2,:]#type2-加购物车
                if len(tem)==0:
                    features['f30'].append(0)
                else:
                    features['f30'].append((last_date-tem.iloc[-1]['date']).days)#最近一次加购物车距今时间
                
                tem=df.ix[(df['type']==2)&(df['cate']==8),:]
                if len(tem)==0:
                    features['f31'].append(0)
                else:
                    features['f31'].append((last_date-tem.iloc[-1]['date']).days)#最近一次加购物车距今时间（cate8）
                
                tem=df.ix[df['type']==3,:]#type3-删除购物车
                if len(tem)==0:
                    features['f32'].append(0)
                else:
                    features['f32'].append((last_date-tem.iloc[-1]['date']).days)#最近一次删除购物车距今时间
                if len(df.ix[(df['type']==2)&(df['cate']==8),:])>len(df.ix[(df['type']==3)&(df['cate']==8),:]):#判断购物车内是否还有商品
                    features['f33'].append(1)
                else:
                    features['f33'].append(0)
                features['f34'].append(len(df.ix[df['type']==2,:]))#历史加入购物车次数   
                features['f35'].append(len(df.ix[df['type']==3,:]))#历史删除购物车次数
            else:#若某些特征没有，将其全部置为999
                features['f5'].append(999)
                features['f6'].append(999)
                for ii in range(7,30):
                    features['f'+str(ii)].append(0)
                features['f30'].append(999)
                features['f31'].append(999)
                features['f32'].append(999)
                for ii in range(33,36):
                    features['f'+str(ii)].append(0)
        else:
            features['f5'].append(999)
            features['f6'].append(999)
            for ii in range(7,30):
                features['f'+str(ii)].append(0)
            features['f30'].append(999)
            features['f31'].append(999)
            features['f32'].append(999)
            for ii in range(33,36):
                features['f'+str(ii)].append(0)
    for k in range(5,36):
        User_s['f'+str(k)]=np.array(features['f'+str(k)])
    return User_s


def get_label(start_date,end_date):#在划分的数据集的时间里，if (type4 != 0) 1 else 0
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    y=[]
    count=0
    for i in User_arr:
        count+=1
        if count%1000==0:
            print(count)
        user_id=str(i)  
        if os.path.exists('C:/Users/gleam/Desktop/user/'+user_id+'.csv'):
            df=pd.read_csv('C:/Users/gleam/Desktop/user/'+user_id+'.csv',encoding="GBK")
            df['date']=pd.to_datetime(df['date'])
            df=df.ix[(df['date']<=end_date)&(df['date']>=start_date)&(df['cate']==8)&(df['type']==4),:]
            if len(df)==0:
                y.append(0)
            else:
                y.append(1) 
        else:
            y.append(0)
    return np.array(y)

##划分数据集
train_feature=data_filter(action,'2016-02-01','2016-04-01')
train_label=data_filter(action,'2016-04-02','2016-04-06')
valid_feature=data_filter(action,'2016-02-01','2016-04-08')
valid_label=data_filter(action,'2016-04-09','2016-04-13')
test_feature=data_filter(action,'2016-02-01','2016-04-15')

#提取特征及label
train_feature=get_feature('2016-04-01')
valid_feature=get_feature('2016-04-08')
test_feature=get_feature('2016-04-15')
train_label=get_label('2016-04-02','2016-04-06')
valid_label=get_label('2016-04-09','2016-04-13')
   
#F11 value-比赛评价指标 
def F11(model,data,y_true):
    y_pred = model.predict(data)
    precision = precision_score(y_true, y_pred)    
    recall = recall_score(y_true, y_pred)
    print('precision:%.4f, recall:%.4f' % (precision, recall))
    return 6*recall*precision/(5*recall+precision)

def F1(model,data,y_true):
    y_pred = model.predict(data)
    precision = precision_score(y_true, y_pred)    
    recall = recall_score(y_true, y_pred)
    print('precision:%.4f, recall:%.4f,F1:%.4f' % (precision, recall,6*recall*precision/(5*recall+precision)))
    print('people_real:%d , people_pred:%d' % (y_true.sum(),y_pred.sum()))
    return 6*recall*precision/(5*recall+precision)
    
   
#读取特征及label    
train_feature=pd.read_csv(r'E:\maching learning\JD\JD\data\feature_hh\train_feature.csv',encoding="GBK")
train_feature = train_feature.set_index('user_id')

valid_feature=pd.read_csv(r'E:\maching learning\JD\JD\data\feature_hh\valid_feature.csv',encoding="GBK")
valid_feature = valid_feature.set_index('user_id')

test_feature=pd.read_csv(r'E:\maching learning\JD\JD\data\feature_hh\test_feature.csv',encoding="GBK")    
test_feature = test_feature.set_index('user_id')


##training model 
max_depth, scale_pos_weight, learning_rate,n_estimators,subsample,colsample_bytree = 5,16,0.02,450,0.6,0.8
model_xgb = xgb.XGBClassifier(max_depth = max_depth, scale_pos_weight = scale_pos_weight,learning_rate = learning_rate,
    n_estimators = n_estimators, objective = 'binary:logistic',subsample = subsample,colsample_bytree = colsample_bytree)
#model.fit(train_feature,train_label)
model.fit(valid_feature.ix[:,:-1],valid_feature['y'])

F11(model,valid_feature.ix[:,:-1],valid_feature['y'])
F11(model,train_feature.ix[:,:-1],train_feature['y'])

#观察预测人数以调节正负样本不平衡的问题
sum(train_label)
sum(valid_label)
sum(model.predict(train_feature))
sum(model.predict(valid_feature))
sum(model.predict(test_feature))

#预测test并输出结果

test_feature['y_pre']=model.predict(test_feature)
test_feature.ix[test_feature.y_pre == 1 ,'y_pre'].to_csv(r'result_6_4.csv',header = True,index = True)

test_feature['y_pre_p']=model.predict_proba(test_feature)[:,1]
test_feature.ix[: ,'y_pre_p'].to_csv(r'result_6_5_p.csv',header = True,index = True)

valid_feature['y_pre_p']=model.predict_proba(valid_feature.ix[:,:-2])[:,1]
valid_feature['y_pre_l']=model.predict(valid_feature.ix[:,:-2])
valid_feature.ix[: ,['y','y_pre_l','y_pre_p']].to_csv(r'result_valid_6_5.csv',header = True,index = True)


scorer = F11(model,data,y_pred)
#CV-xgb
from sklearn.cross_validation import cross_val_score

max_depth, scale_pos_weight, learning_rate,n_estimators,subsample,colsample_bytree = 5,11,0.02,450,0.6,0.8
model_xgb = xgb.XGBClassifier(max_depth = max_depth, scale_pos_weight = scale_pos_weight,learning_rate = learning_rate,
    n_estimators = n_estimators, objective = 'binary:logistic',subsample = subsample,colsample_bytree = colsample_bytree)

result_xgb = cross_val_score(model_xgb,
valid_feature.ix[:,:-1],
valid_feature['y'],
cv = 5,
scoring = F1).mean()

#cv-lr
f2_fea = pd.get_dummies(valid_feature['f2'],prefix = 'f2')
f3_fea = pd.get_dummies(valid_feature['f3'],prefix = 'f3')
valid_feature_lr = valid_feature.drop(['f2','f3'],axis = 1)
valid_feature_lr = pd.concat([f2_fea,f3_fea,valid_feature_lr],axis = 1)
     
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(class_weight= {1:13})

result_lr = cross_val_score(model_lr,
valid_feature_lr.ix[:,:-1],
valid_feature_lr['y'],
cv = 5,
scoring = F1).mean()

#cv-rf
from sklearn.ensemble import RandomForestClassifier
n_estimators,max_depth ,class_weight= 300,7,{1:15}
model_rf = RandomForestClassifier(n_estimators=n_estimators,max_depth = max_depth,class_weight = class_weight)

result_rf = cross_val_score(model_rf,
valid_feature.ix[:,:-1].fillna(999),
valid_feature['y'],
cv = 5,
scoring = F1).mean()












