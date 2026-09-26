import numpy as np

def expanding_splits(n, min_train, test_size, embargo=0):
    if min_train<=0 or test_size<=0: raise ValueError('min_train and test_size must be positive')
    end=min_train
    while end+embargo+test_size<=n:
        train=np.arange(end); test=np.arange(end+embargo,end+embargo+test_size)
        yield train,test; end+=test_size

def year_splits(years, min_train_years=8, test_years=1, final_years=2):
    ys=sorted(set(int(y) for y in years))
    if len(ys)<min_train_years+test_years+final_years: return
    research=ys[:-final_years] if final_years else ys
    for i in range(min_train_years,len(research),test_years):
        tr=research[:i]; te=research[i:i+test_years]
        if te: yield tr,te

def final_holdout(years, final_years=2):
    ys=sorted(set(int(y) for y in years)); return ys[-final_years:] if final_years else []
