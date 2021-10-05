# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

## Question 0
### (a)
'''
The code above accomplishes a task that find all tuples from a list whose last element is the largest 
in each group, which is composed by tuples whose first elements are the same.If both first value and 
last value are the same for some tuples,then print them all.

'''

### (b)
''' 
Code Review:

1. This code has 2 bugs.It would be better fix all bugs before submission.
    The first bug is that the indent of the fifth row is not right.The loop of parameter 'n' should not 
indent.The second bug is that the index 3 of the list should be 2. 

2. The genral style of this code is good, but still can improve.
    The length of every row of code controls well. Yet it would be a greater style if an annotation 
of function that code realizes is added above code.   
   
3. The code structure is good but not that easy to understand.
    The 'if' condition language in the code is too long to read. In this case annotations of some 
variables and sentences were suggested to add to be better for reading.. 

4. The code truly realizes the function mentioned in question(a), but the efficiency could be 
improved by simplifying the second 'for' loop. The second 'for' could iterate from the value 
of the first iteration m on rather than from 1. A kind suggestion is that change the range 
of the second 'for' loop in the code for higher efficiency.

'''


## Question 1
import numpy as np
def tuples(n, low=0, high=10, k=5):
    """
    This function aims to generate a random list of n k-tuples containing integers
    ranging from low to high and output them.
    
    Paramaters
    ----------
    n : integer,
      This parameter represents the number of tuples.
    low : integer,
      This parameter represents the smallest number of tuple.
    high : integer,
      This parameter represents the largest number of tuple.
    k : integer,
      This parameter represents the number of elements of each tuple,
    
     Returns
     -------
     n k-tuples
    """
    try:
        high - low > 0 
        l = [0] * n # the list used to contain n k-tuples
        for i in range(0, n):
            l[i] = tuple(np.random.randint(low, high + 1, k)) # np.random.randint(a,b) generates [a, b-1]
            assert type(l[i]) == tuple, 'The generated element is not a tuple!'
        assert type(l) == list, 'Output is not a list of tuples!' 
        return l
    except ValueError:
        return('Value Error: The value of parameter <high> must be larger than parameter <low> ! ')
## test
tuples(4, low=3, high=10, k=9)


## Question 2
### (a) 
import time
from tabulate import tabulate
def compare(sample_list, first, last):
    """
    This function aims to find tuples with each unique first element.For equal first element tuples,
    find the one with largest last element.
    
    Parameters
    ----------
    sample_list : list,
      This parameter represents the a list of several tuples.
    first : integer,
      This parameter represents the first index of each tuple
    last : integer,
      This parameter represents the last index of each tuple
    
    Returns
    -------
    res : list,
      A list of finding tuples
    
    """
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        # source code here has 4 indents, I asjust it myself.
        for n in range(len(sample_list)):
            if (sample_list[m][first] == sample_list[n][first] and
                # here index should be 2 rather than 3
                    sample_list[m][last] != sample_list[n][last]):
                li.append(sample_list[n])
        op.append(sorted(li, key = lambda dd : dd[2], reverse = True)[0])
    res = list(set(op))
    return res
# test
sample_list = [(1, 3, 5), (0, 1, 2), (5, 6, 2), (0, 9, 7),(1, 4, 5)]
compare(sample_list, 0, 8)


### (b)
def imp_compare(sample_list, first, last):
    """
    This function aims to find tuples with each unique first element.For equal first element tuples,
    find the one with largest last element.
    
    Parameters
    ----------
    sample_list : list,
      This parameter represents the a list of several tuples.
    first : integer,
      This parameter represents the first index of each tuple
    last : integer,
      This parameter represents the last index of each tuple
    
    Returns
    -------
    res : list,
      A list of finding tuples
    
    """
    # a vector to contain unique first value among all tuples
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        # the range is lower
        for n in range(m,len(sample_list)):
            # it is not necessary to compare whether the last element is identical,
            # because when last elements are equal, 'sorted ' function will choose the 
            # one with smaller index. Then because in loop m, list 'li' appends every 
            # element during the whole iteration. So no element will be ignored in this case.
            if (sample_list[m][first] == sample_list[n][first]):
                li.append(sample_list[n])
        op.append(sorted(li, key = lambda dd: dd[2], reverse = True)[0])
    res = list(set(op))
    return res
# test
sample_list = [(1, 3, 5), (0, 1, 2), (5, 6, 2), (0, 9, 7), (1, 4, 5)]
imp_compare(sample_list, 0, 8)


### (c)
def dict_compare(sample_list, first, last):
    """
    This function aims to find tuples with each unique first element.For equal first element tuples,
    find the one with largest last element.
    
    Parameters
    ----------
    sample_list : list,
      This parameter represents the a list of several tuples.
    first : integer,
      This parameter represents the first index of each tuple
    last : integer,
      This parameter represents the last index of each tuple
        
    Returns
    -------
    op : list,
      A list of finding tuples
    """
    dic = {} # Dic parameter represents the dictionary to store the tuples.
    f_element = []  # a list to contain first values
    op = []  # output list
    for m in range(len(sample_list)):
        f_value = sample_list[m][first] # first element of tuple
        #  Create keys and values of dictionary.
        #  Tuples with the same first element are stored within the same key.
        if f_value not in f_element:
            f_element.append(f_value)
            dic[str(f_value)] = [sample_list[m]]
        else:
            dic[str(f_value)].append(sample_list[m]) 
    # loop dictionary and find the tuple with the largest last element in each key.
    # Then load them in list op in each iteration. When loop finished, op is the result.
    for n in dic.keys():
        m_last = max([i[last] for i in dic[n]])
        objects = [j for j in dic[n] if j[last] == m_last]
        op.extend(objects)
    return op        
 # test
sample_list = [(1, 3, 5), (0, 1, 2), (5, 6, 2), (0, 9, 7), (1, 4, 5)]
dict_compare(sample_list, 0, 6)              

# +
### (d)
"""
Monte carlo method comments:
    From monte carlo result, when n are above 10, dictionary method is much more efficient than former two.
"""
# number of tuples in the list
result = pd.DataFrame(columns = ['n value', 'k value', 'compare median', 'improve median', 'dictionary median'])
# count nrows of result
row_count = 0
for n in range(4, 14, 2):
    for k in range(5, 17, 2):
        # lists to contain time for each function
        row_count = row_count + 1
        ori_t = []
        imp_t = []
        dic_t = []
        for i in range(0,10):
            sample_list = tuples(n=n, low=0, high=10, k=k)
            # time for function compare
            ot_start = time.perf_counter()
            compare(sample_list, 0, k-1)
            ot_end = time.perf_counter()
            ori_t.append(ot_end - ot_start)
            # time for function imp_compare
            it_start = time.perf_counter()
            imp_compare(sample_list, 0, k-1)
            it_end = time.perf_counter()
            imp_t.append(it_end - it_start)
            # time for function dict_compare
            dt_start = time.perf_counter()
            dict_compare(sample_list, 0, k-1)
            dt_end = time.perf_counter()
            dic_t.append(dt_end - dt_start)
        result.loc[row_count] = [n, k, np.median(ori_t), np.median(imp_t), np.median(dic_t)]

display(result)      
# -

## Question 3
import pandas as pd
import re
import pickle
### (a)
## I choose 4 datasets: NHANES 2011-2012 | 2013-2014 | 2015-2016 | 2017-2018
demogr_11_12 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT')
demogr_13_14 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT')
demogr_15_16 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT')
demogr_17_18 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT')

# +
columns = ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'RIDSTATR', 'SDMVPSU', \
           'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']
demogr_12 = demogr_11_12.loc[:, columns]
demogr_12['type'] = '11-12'
# From the expression of demographic data, variable 'SEQN' is string type,
# variables 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'RIDSTATR', 'SDMVPSU' are categorical,
# variables 'RIDAGEYR' and 'SDMVSTRA' are integers.
# so transform types of variables above.
cate_col = ['RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'RIDSTATR', 'SDMVPSU']
demogr_12 = demogr_12.astype({'SEQN' : 'str', 'RIDAGEYR' : 'int', 'SDMVSTRA' : 'int'}) 
for i in cate_col:
    demogr_12.loc[:, i] = demogr_12.astype({i : 'category'})
# rename columns to make them readable    
demogr_12 = demogr_12.rename(columns = {'SEQN' : 'id','RIDAGEYR' : 'age', 'RIDRETH3' : 'race', \
    'DMDEDUC2' : 'education', 'DMDMARTL' : 'marital Status','RIDSTATR' : 'examine status',\
    'SDMVPSU' : 'masked pseudo-psu','SDMVSTRA' : 'masked pseudo-stratum',\
    'WTMEC2YR' : 'interview & mec','WTINT2YR' : 'interview'})

demogr_34 = demogr_13_14.loc[:, columns]
demogr_34['type'] = '13-14'
demogr_34 = demogr_34.astype({'SEQN' : 'str', 'RIDAGEYR' : 'int', 'SDMVSTRA' : 'int'}) 
for i in cate_col:
    demogr_34.loc[:, i] = demogr_34.astype({i : 'category'})
demogr_34 = demogr_34.rename(columns = {'SEQN' : 'id','RIDAGEYR' : 'age', 'RIDRETH3' : 'race', \
    'DMDEDUC2' : 'education', 'DMDMARTL' : 'marital Status','RIDSTATR' : 'examine status',\
    'SDMVPSU' : 'masked pseudo-psu','SDMVSTRA' : 'masked pseudo-stratum',\
    'WTMEC2YR' : 'interview & mec','WTINT2YR' : 'interview'})

demogr_56 = demogr_15_16.loc[:, columns]
demogr_56['type'] = '15-16'
demogr_56 = demogr_56.astype({'SEQN' : 'str', 'RIDAGEYR' : 'int', 'SDMVSTRA' : 'int'}) 
for i in cate_col:
    demogr_56.loc[:, i] = demogr_56.astype({i : 'category'})
demogr_56 = demogr_56.rename(columns = {'SEQN' : 'id','RIDAGEYR' : 'age', 'RIDRETH3' : 'race', \
    'DMDEDUC2' : 'education', 'DMDMARTL' : 'marital Status','RIDSTATR' : 'examine status',\
    'SDMVPSU' : 'masked pseudo-psu','SDMVSTRA' : 'masked pseudo-stratum',\
    'WTMEC2YR' : 'interview & mec','WTINT2YR' : 'interview'})

demogr_78 = demogr_17_18.loc[:, columns]
demogr_78['type'] = '17-18'
demogr_78 = demogr_78.astype({'SEQN' : 'str', 'RIDAGEYR' : 'int', 'SDMVSTRA' : 'int'}) 
for i in cate_col:
    demogr_78.loc[:, i] = demogr_78.astype({i : 'category'})
demogr_78 = demogr_78.rename(columns = {'SEQN' : 'id','RIDAGEYR' : 'age', 'RIDRETH3' : 'race', \
    'DMDEDUC2' : 'education', 'DMDMARTL' : 'marital Status','RIDSTATR' : 'examine status',\
    'SDMVPSU' : 'masked pseudo-psu','SDMVSTRA' : 'masked pseudo-stratum',\
    'WTMEC2YR' : 'interview & mec','WTINT2YR' : 'interview'})

# -

# output pickle document
demogr = pd.concat([demogr_12, demogr_34, demogr_56, demogr_78])
demogr.to_pickle('./demogr_total.pkl')

### (b)
oral_11_12 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHXDEN_G.XPT')
oral_13_14 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHXDEN_H.XPT')
oral_15_16 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHXDEN_I.XPT')
oral_17_18 = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXDEN_J.XPT')               

# +
columns = ['SEQN', 'OHDDESTS']
col_list = oral_11_12.columns.tolist()

# match tooth counts (OHXxxTC)
pattern_1 = re.compile(r'OHX[\d][\d]TC')
m_1 = [pattern_1.search(i) for i in col_list]  # loop all columns and match in the columns 
m_1 = [i for i in m_1 if i != None]  # extract not None value
m_1 = [i.group(0) for i in m_1]  # extract match value
columns.extend(m_1) # add matched values into column list.
# print(len(columns)) Here 34 columns in total

# match coronal cavities (OHXxxCTC) in the columns
pattern_2 = re.compile(r'OHX[\d][\d]CTC')
m_2 = [pattern_2.search(i) for i in col_list]
m_2 = [i for i in m_2 if i != None]
m_2 = [i.group(0) for i in m_2]
columns.extend(m_2)
# print(len(columns)) Here 62 columns in total

oral_12 = oral_11_12.loc[:, columns]
oral_12['type'] = '11-12'
oral_34 = oral_13_14.loc[:, columns]
oral_34['type'] = '13-14'
oral_56 = oral_15_16.loc[:, columns]
oral_56['type'] = '15-16'
oral_78 = oral_17_18.loc[:, columns]
oral_78['type'] = '17-18'

## column type transform:
## columns (OHXxxCTC) are read as byte type.
## They are supposed to be string type.
oral_12 = oral_12.astype({'SEQN' : 'str','OHDDESTS' : 'category'}) 
oral_34 = oral_34.astype({'SEQN' : 'str','OHDDESTS' : 'category'}) 
oral_56 = oral_56.astype({'SEQN' : 'str','OHDDESTS' : 'category'})
oral_78 = oral_78.astype({'SEQN' : 'str','OHDDESTS' : 'category'}) 
# iterate column indice of OHXxxCTC
for i in range(34, len(columns)):
    oral_12.iloc[:, i] = oral_12.iloc[:, i].map(lambda x : str(x, encoding = 'utf-8'))
    oral_34.iloc[:, i] = oral_34.iloc[:, i].map(lambda x : str(x, encoding = 'utf-8'))
    oral_56.iloc[:, i] = oral_56.iloc[:, i].map(lambda x : str(x, encoding = 'utf-8'))
    oral_78.iloc[:, i] = oral_78.iloc[:, i].map(lambda x : str(x, encoding = 'utf-8'))
# rename columns
oral_12 = oral_12.rename(columns = {'SEQN' : 'id', 'OHDDESTS' : 'status code'}) 
oral_12.columns = [i.lower() for i in list(oral_12.columns)]
oral_34 = oral_34.rename(columns = {'SEQN' : 'id', 'OHDDESTS' : 'status code'}) 
oral_34.columns = [i.lower() for i in list(oral_12.columns)]
oral_56 = oral_56.rename(columns = {'SEQN' : 'id', 'OHDDESTS' : 'status code'}) 
oral_56.columns = [i.lower() for i in list(oral_12.columns)]
oral_78 = oral_78.rename(columns = {'SEQN' : 'id', 'OHDDESTS' : 'status code'}) 
oral_78.columns = [i.lower() for i in list(oral_12.columns)]
# -

## output pickle document
oral = pd.concat([oral_12, oral_34, oral_56, oral_78])
oral.to_pickle('./oral_dentition_total.pkl')


### (c)
## Case of demographic data is:  39156
## Case of Oral dentition data is:  35909
print('Case of demographic data is: ', demogr.shape[0])
print('Case of Oral dentition data is: ', oral.shape[0])
