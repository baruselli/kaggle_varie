import numpy as np
from sklearn import *

def RMSLE(y, pred):
    return metrics.mean_squared_error(np.log(y+1), np.log(pred+1))**0.5
def RMSE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5
    
    
def bin_enc(df,cols_to_enc):
    import category_encoders as ce
    encoder = ce.BinaryEncoder(cols=cols_to_enc)
    enc=encoder.fit_transform(df)
    return enc
    
    
def train_eval_test(df,cut_date,target="unit_sales",date="date"):
    df_train=df[df[target].notnull()][df[date]<cut_date]        #train
    df_eval= df[df[target].notnull()][df[date]>= cut_date]      #test (I have data to check)
    df_test= df[df[target].isnull()]                                   #eval (for submission only, no data to check)
    return(df_train,df_eval,df_test)
    
    
def xy_train_test(df_train,df_test,y,drop=[]):
        X_train = df_train.drop(drop,axis=1).as_matrix()
        X_test =  df_test.drop(drop,axis=1).as_matrix()
        y_train = df_train[y].values
        y_test =  df_test[y].values
        
        #return(X_train,X_test,y_train,y_test)              
        return(X_train.astype("float32"),X_test.astype("float32"),y_train.astype("float32"),y_test.astype("float32"))              


def test(df_train,df_test,regr,y,drop):
    (X_train,X_test,y_train,y_test) = xy_train_test(df_train,df_test,y,drop)
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred = regr.predict(X_test)
    y_pred=np.maximum(0,y_pred)
    y_pred_train=np.maximum(0,y_pred_train)
    error=RMSE(y_test, y_pred)
    error_train=RMSE(y_train, y_pred_train)
    print(error,error_train)
    return error,error_train

    
    
def make_csv(df_train,df_eval,df_test,regr,y,file,drop=[],columns=[]):
    df_train_tot=df_train.append(df_eval)
    #print(df_train_tot.columns)
    #print(df_eval.columns)
    X_train = df_train_tot.drop(drop,axis=1).as_matrix()
    X_test =  df_test.drop(drop,axis=1).as_matrix()
    # Split the targets into training/testing sets
    y_train = df_train_tot[y].values
    print(X_train.shape,y_train.shape)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    y_pred=np.maximum(0,y_pred)
    y_pred=np.expm1(y_pred)
    df_test[y]=y_pred
    #df_test["id"]=df_eval["air_store_id"].map(str)+"_"+df_eval["visit_date"].dt.strftime('%Y-%m-%d')
    #df_sub=df_eval[["id","visitors"]]
    df_test.to_csv(file,index=False,columns=columns)
    
def lognuniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))