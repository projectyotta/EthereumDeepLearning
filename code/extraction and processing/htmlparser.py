
# coding: utf-8

# In[469]:

import requests
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import networkx as nx


# In[475]:

# read the html file 
x=pd.read_html("C:/Users/raosa/Desktop/PROJECTS/blockchain/ethereum/html/block_test1.html")
# this prints out the info with the incomplete to and from address  
x=x[1]


# we need to get the full to and from address 
soup = BeautifulSoup(open("C:/Users/raosa/Desktop/PROJECTS/blockchain/ethereum/html/block_test1.html"), "html.parser")
soup1 = list(soup.children)[1]
soup2 = list(soup1.children)[1]
soup3 = list(soup2.children)[1]
soup4 = list(soup3.children)[2]
soup5 = list(soup4.children)[0]
soup6 = list(soup5.children)[4]
soup7 = list(soup6.children)[0]
soup8 = list(soup7.children)[0]


table_body=soup8.find('tbody')
rows = table_body.find_all('tr')
a=[]
b=[]
for i in range(0,len(rows)):
    a.append(rows[i].find_all('a')[0])
    b.append(rows[i].find_all('a')[1])

x['From'] = a
x['To'] = b
x['From'] = x['From'].astype(str)
x['To'] = x['To'].astype(str)
substring1 = '<a href="/'
substring2 = '">'


for i in range(0,len(x)):
      x['From'][i] = x['From'][i][(x['From'][i].index(substring1)+len(substring1)):x['From'][i].index(substring2)]
      x['To'][i] = x['To'][i][(x['To'][i].index(substring1)+len(substring1)):x['To'][i].index(substring2)]
      x['From'][i] = x['From'][i].split("/",1)[1]
      x['To'][i] = x['To'][i].split("/",1)[1]
# output as a csv file 
x.to_csv('transactions.csv')
# for row in rows:
#     cols=row.find_all('td')
#     cols=[x.text.strip() for x in cols]
#     print(cols)


# In[484]:

# convert to adjacency matrix 
# note , a third value can be attached in the below line called weights . This can be the ETH value. 
G=nx.from_pandas_dataframe(x,'From','To')
Adjtraining = nx.adjacency_matrix(G)
adj = Adjtraining.todense()
print(adj[0])


# In[455]:
# read the html file 
x=pd.read_html("C:/Users/raosa/Desktop/PROJECTS/blockchain/ethereum/html/block_test1.html")
x=x[0]
names = []
values = []
names.append(x[0])
names.append(x[1])
y = pd.DataFrame(columns=names[0])
y.loc[0] = np.array(names[1])
y.to_csv('blockinfo.csv')

