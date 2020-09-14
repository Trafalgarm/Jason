import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# color = sns.color_palette()  #创建调色板
# pd.set_option('precision',3)  #设置精度
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.max_columns=777
# df=pd.read_csv(r'C:\Users\76364\Desktop\wine\winequality-white.csv')
# # print(df.head())
# # print('+'*20)
# # print(df.info())
# # print('+'*20)
# # print(df.describe(include='object'))
# col=df.columns.str.split(';').tolist()
# for i in range(len(col[0])):
#     col[0][i]=col[0][i].strip('"')
# col=col[0]
#
# value = []
# all_data = []
# for i in range(len(df.index.tolist())):
#     value = df.iloc[i].str.split(';').tolist()
#     value = value[0]
#     all_data.append(value)
# # print(all_data)
#
# wine1=pd.DataFrame(data=all_data,columns=col,dtype=float)
# print(wine1.head())
#
# plt.style.use('ggplot')
# figure=plt.figure(figsize=(10,6))
# for i in range(12):
#     plt.subplot(2,6,i+1)
#     sns.boxplot(wine1[col[i]],orient='v')
#     plt.ylabel(col[i],fontsize=12)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10,8))
# for i in range(12):
#     plt.subplot(4,3,i+1)
#     wine1[col[i]].hist(bins=100,color='r')
#     plt.xlabel(col[i],fontsize=12)
#     plt.ylabel('Frequency')
# plt.show()
#
# wine1['total acid'] = wine1['fixed acidity']+wine1['volatile acidity']+wine1['citric acid']
# plt.figure(figsize=(8,3))
# plt.subplot(121)
# plt.hist(wine1['total acid'],bins=50,color='b')
# plt.xlabel('total acid')
# plt.ylabel('Frequency')
# plt.subplot(122)
# plt.hist(np.log(wine1['total acid']),bins=50,color='b')
# plt.xlabel('log(total acid)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()

# wine1['sweetness'] = pd.cut(wine1['residual sugar'],bins=[0,4,12,45],labels=["dry","medium dry","semi-sweet"])
# plt.figure(figsize=(5,3))
# wine1['sweetness'].value_counts().plot(kind='bar',color='b')
# plt.xticks(rotation=0)
# plt.xlabel('sweetness',fontsize=12)
# plt.ylabel('Frequency',fontsize=12)
# plt.tight_layout()
# plt.show()
# print(wine1['sweetness'])

# sns.set_style('ticks')
# sns.set_context('notebook',font_scale=1.1)
# colnm = wine1.columns.tolist()[:11]+['total acid']
# plt.figure(figsize=(10,8))
# for i in range(12):
#     plt.subplot(4,3,i+1)
#     sns.barplot(x='quality',y=colnm[i],data=wine1,color='g',width=0.6)
# plt.tight_layout()
# plt.show()

# sns.set_style('dark')
# plt.figure(figsize=(10,8))
# colnm=wine1.columns.tolist()[:11]+['total acid','quality']
# mcorr = wine1[colnm].corr()
# mask = np.zeros_like(mcorr)
# mask[np.triu_indices_from(mask)] = True
# cmap = sns.diverging_palette(220,10,as_cmap=True)
# g = sns.heatmap(mcorr,mask=mask,cmap=cmap,square=True,annot=True,fmt='0.2f')
# plt.show()

# sns.set_style('ticks')
# sns.set_context("notebook",font_scale=1.4)
# plt.figure(figsize=(6,4))
# #画出双变量的散点图，然后以y~x拟合回归方程和预测值95%置信区间并将其画出。
# sns.regplot(x='density',y='alcohol',data=wine1,scatter_kws={'s':100},color=color[1])
# plt.xlim(0.989, 1.005)
# plt.ylim(7,16)
# plt.show()

# acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide',
#                    'sulphates', 'total acid']
# plt.figure(figsize = (10,6))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     sns.regplot(x='pH', y = acidity_related[i], data = wine1, scatter_kws = {'s':10}, color = color[1])
# plt.tight_layout()
# plt.show()

# plt.style.use('ggplot')
#  sns.lmplot(x = 'alcohol', y = 'volatile acidity', hue = 'quality',
#             data = wine1, fit_reg = False, scatter_kws={'s':10}, height = 5)
# sns.lmplot(x = 'alcohol', y = 'volatile acidity', col='quality', hue = 'quality',
#            data = wine1,fit_reg = False, size = 3,  aspect = 0.9, col_wrap=3,
#            scatter_kws={'s':20})
# plt.show()

# sns.set_style('ticks')
# sns.set_context("notebook", font_scale= 1.4)
# plt.figure(figsize=(6,5))
# cm = plt.cm.get_cmap('RdBu')
# sc = plt.scatter(wine1['fixed acidity'], wine1['citric acid'], c=wine1['pH'], vmin=2.6, vmax=4, s=15, cmap=cm)
# bar = plt.colorbar(sc)
# bar.set_label('pH', rotation = 0)
# plt.xlabel('fixed acidity')
# plt.ylabel('citric acid')
# # plt.xlim(4,18)
# # plt.ylim(0,1)
# plt.show()

# color = sns.color_palette()  #创建调色板
# pd.set_option('precision',3)  #设置精度
# plt.rcParams['font.sans-serif']=['SimHei']
# pd.options.display.max_columns=777
# df=pd.read_csv(r'C:\Users\76364\Desktop\wine\zijie.csv',dtype=str)
# # print(df.head())
# # print('+'*20)
# # print(df.info())
# # print('+'*20)
# # print(df.describe(include='object'))
#
# col = df.columns.tolist()
# print(col)
# # print(df.index)
# print('-'*100)
# new = df.drop(columns=[' 真实姓名的MD5编码',' 是否被锁定在其他岗位',' 进入新阶段的时间'])
# print(df.head())
# num = new[new[ ' 岗位'].str.contains('产品经理')].reset_index(drop=True)
# num = num[num[ ' 岗位'].str.contains('企业应用')].reset_index(drop=True)
# print(num)

# data = pd.read_csv(r'C:\Users\76364\Desktop\wine\mountains.csv')
# data.columns = ['heighest','tall','mountain']
# print(data.head())
# data['tall']=data['tall'].fillna(0).astype(int)
# data['tall']=data['tall'].apply(lambda x: 200 if '1'in str(x) else 160)
# x = data[data==160]
# print(x)

xls = pd.read_excel(r'C:\Users\76364\Desktop\wine\朝阳医院2018年销售数据.xlsx',dtype='object')
print(xls.head(),xls.dtypes)
subsale = xls.loc[0:4,'购药时间':'销售数量']
xls.rename(columns={'购药时间':'销售时间'},inplace=True)
print(xls.columns)
print(xls.shape)
sales = xls.dropna(subset=['销售时间','社保卡号'],how='any').reset_index(drop=True)
sales = pd.DataFrame(sales)
print(sales.shape)
sales['销售数量'] = sales['销售数量'].astype(float)
print(sales['销售数量'].dtype)
sales['应收金额'] = sales['应收金额'].astype('float')
sales['实收金额'] = sales['实收金额'].astype('float')
def splitsales(x):
    timelist=[]
    for i in x:
        datestr = i.split(' ')[0]
        timelist.append(datestr)
    time = pd.Series(timelist)
    return time
y=sales.loc[:,'销售时间']
datesales = splitsales(y)
sales.loc[:,'销售时间']=pd.to_datetime(datesales,format='%Y-%m-%d',errors='coerce')
print(sales.head())
sales = sales.dropna(subset=['销售时间','社保卡号'],how='any')
sales= sales.sort_values(by='销售时间',ascending=True)
sales = sales.reset_index(drop=True)
print(sales.head())
print(sales.shape)
sales = sales.loc[sales['销售数量']>0]
sales=sales.reset_index(drop=True)
print(sales.shape)
kdi= sales.drop_duplicates(subset=['销售时间', '社保卡号'])
total=kdi.shape[0]
print(total)
kdi=kdi.sort_values(by='销售时间',ascending=True)
kdi=kdi.reset_index(drop=True)
startime= kdi.loc[0,'销售时间']
endtime=kdi.loc[total-1,'销售时间']
days=(endtime-startime).days
months=days//30
print(months)
kdq=total//months
print(kdq)
totalmoney=sales['实收金额'].sum()
monthmoney=totalmoney/months
pct=totalmoney/total
print(monthmoney)
print(sales.head())
group=sales
group.set_index(group['销售时间'],inplace=True)
gb=group.groupby(group.index.month)
monthgp=gb.sum()
print(monthgp)