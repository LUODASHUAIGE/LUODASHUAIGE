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
