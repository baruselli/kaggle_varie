import numpy as np
from sklearn import *

def RMSLE(y, pred):
    return metrics.mean_squared_error(np.log(y+1), np.log(pred+1))**0.5
def RMSE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5
    
    
#def bin_enc(df,cols_to_enc):
#    import category_encoders as ce
#    print("fit")
#    encoder = ce.BinaryEncoder(cols=cols_to_enc,verbose=10).fit(df)
#    print("transform")
#    enc=encoder.transform(df)
#    return enc
    
    

def bin_enc(df_in,cols_to_enc,verbose=1, drop_original=True,copy=False, shuffle=False,seed=0,ordinal_only=False):
    """Converts categorical/integer columns into binary representation, using numpy bit-wise operations
       Input:
        df_in: dataframe to be manipulated
        cols_to_enc: list of columns to encode (categorical or integer)
        verbose: how much stuff to print
        drop_original: delete original columns given in cols_to_enc
        copy: if True, return a new dataframe; if False, works on the input dataframe df_in
        shuffle: if True, shuffle categories from the standard choice by pandas
        seed: if shuffle, seed for shuffling
        ordinal_only: if True, encodes categories as integers, and stops
         """
    import numpy as np
    #if necessary, copies df, otherwise just aliases it
    if (copy): df=df_in.copy()
    else:      df=df_in
    #set seed for shuffling
    if(shuffle):
        import random
        random.seed(seed)
        if(verbose>0): print("Shuffling with seed ", seed)
    #loop over columns to encode
    for col in cols_to_enc:
        if(verbose>0): print("reading col "+col)
        #try to convert into category
        try:    df[col]=df[col].astype("category")
        except: print ("WARNING: problem converting in category")
        # shuffle categories if requested
        if(shuffle):
             cats=df[col].cat
             cats_list=list(cats.categories)
             new_list=random.sample(cats_list,len(cats_list))
             if(verbose>2): print("  shuffle with list", new_list)
             cats.reorder_categories(new_list,inplace=True)
        #try to get category codes
        try:    df[col]= df[col].cat.codes
        except: print ("WARNING: problem getting category codes")
        #if only ordinal encoding, stop here
        if(ordinal_only): return df
        ##starts the binary encoding process
        #maximum number to encode
        cat_max=df[col].max()
        if(verbose>0): print("  maximum category index ",cat_max)
        #number of required bits
        from math import ceil
        try:
            n_bits=ceil(np.log2(cat_max+1))
        except:
            print("WARNING: cannot compute number of bits, setting to 1 (maybe just one category is present)")
            n_bits=1
        if(verbose>0): print("  number of bits ",n_bits)
        #creates temporary np arrays
        array=df[col].values
        array_tmp=np.zeros((len(array), n_bits))
        #loop over bits
        for bit in range(n_bits):
            new_col=str(col)+"_"+str(bit)
            if(verbose>1): print("  creating new column" , new_col)
            #compute bit for each integer, cryptic but it does the job (fast)
            #taken from https://stackoverflow.com/a/20643178
            array_tmp[:,bit]=1 & array[:] >> bit
            #creates a new column for every bit and fills it with the temp array
            df[new_col]=array_tmp[:,bit]
            #bit values are just 0/1, so int8 is enough
            df[new_col] = df[new_col].astype("int8")
        #deletes temporary arrays
        del(array_tmp)
        del(array)
        # if necessary, drops original columns        
        if(drop_original): df.drop(labels=col,axis=1,inplace=True)
    if (verbose>0): print ("Done!")
    return df



def train_eval_test(df,cut_date,target="unit_sales",date="date"):
    df_train=df[df[target].notnull()][df[date]<cut_date]        #train
    df_eval= df[df[target].notnull()][df[date]>= cut_date]      #test (I have data to check)
    df_test= df[df[target].isnull()]                                   #eval (for submission only, no data to check)
    return(df_train,df_eval,df_test)
    
    
def xy_train_test(df_train,df_test,y,drop=[]):
        drop.append(y)
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
    drop.append(y)
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
    df_test.to_csv(file,index=False,columns=columns,float_format='%.5f')
    
def lognuniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))
