#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import seaborn as sns


# In[2]:


df = pd.read_csv("D:\\Documents\\BSCS 7D\\Data science\\data.csv", encoding='latin-1')


# In[3]:


df.head()


# In[4]:


df.shape


# <font size="39"><b>Data Cleaning</b></font>

# In[5]:


def data_cleaning(data):
    #we removed spaces before adn after decription
    data['Description'] = data['Description'].str.strip()
    #duplicate invoice remove kardi
    data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    #invoiceNo type convert to string
    data['InvoiceNo'] = data['InvoiceNo'].astype('str')
    #cancelled transaction removed (C mean invoice wasnt generated)
    data = data[~data['InvoiceNo'].str.contains('C')]
    #Drop extra columns
    data = data.drop(columns=['StockCode', 'InvoiceDate', 'UnitPrice', 'CustomerID'])
    return data

df = data_cleaning(df)


# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


#removed null
df = df.dropna()


# In[9]:


#updated indexes
df = df.reset_index()   
df.drop(["index"], axis=1, inplace=True)


# In[10]:


df.head()


# In[11]:


df.dtypes


# In[12]:


df.describe()


# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df['Country'].value_counts()


# In[16]:


top3 = df["Country"].value_counts().head(3)

plt.figure(figsize=[10, 6])
top3.plot(kind='bar', color='skyblue')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.title('Top 10 Countries by Number of Transactions')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()


# In[17]:


most_sold = df['Description'].value_counts().head(10)

plt.figure(figsize=[10, 6])
most_sold.plot(kind='bar', color='skyblue')
plt.xlabel('Items')
plt.ylabel('Number of Items sold')
plt.title('Top 10 Items Sold')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()


# <font size="39"><b>Data Training</b></font>

# In[18]:


def pivot_column(df, country):
    basket = (df[df['Country'] == country]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))

    def encode_data(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket = basket.applymap(encode_data)
    basket.drop('POSTAGE', inplace=True, axis=1)
    basket = basket[(basket > 0).sum(axis=1) >= 2]
    return basket


uk_basket = pivot_column(df, 'United Kingdom')
germany_basket = pivot_column(df, 'Germany')
france_basket = pivot_column(df, 'France')


uk_frequent_itemsets = apriori(uk_basket, min_support=0.03, use_colnames=True)
germany_frequent_itemsets = apriori(germany_basket, min_support=0.03, use_colnames=True)
france_frequent_itemsets = apriori(france_basket, min_support=0.03, use_colnames=True)


uk_rules = association_rules(uk_frequent_itemsets, metric='lift', min_threshold=1).sort_values("lift", ascending=False).reset_index(drop=True)
germany_rules = association_rules(germany_frequent_itemsets, metric='lift', min_threshold=1).sort_values("lift", ascending=False).reset_index(drop=True)
france_rules = association_rules(france_frequent_itemsets, metric='lift', min_threshold=1).sort_values("lift", ascending=False).reset_index(drop=True)


# <p style="font-size:41px"><b>UK Basket</b></p>

# <p style="font-size:18px">UK Basket Generated Rules</p>

# In[19]:


uk_rules.head(100)


# In[20]:


uk_rules.sort_values('confidence', ascending=False)


# In[21]:


uk_rules[ (uk_rules['lift'] >= 3) & (uk_rules['confidence'] >= 0.5) & (uk_rules['support'] >= 0.03)]


# <p style="font-size:18px">UK Basket Predicted Visualization</p>

# In[22]:


top_rules = uk_rules.nlargest(10, 'lift') 

rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(12, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['lift'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by Lift in UK')
plt.show()


# In[23]:


top_rules = uk_rules.nlargest(10, 'confidence')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(14, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['confidence'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by confidence in UK')
plt.show()


# In[24]:


top_rules = uk_rules.nlargest(10, 'support')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(14, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['support'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by support in UK')
plt.show()



# In[25]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=uk_rules['support'],
    y=uk_rules['confidence'],
    z=uk_rules['lift'],
    mode='markers',
    marker=dict(color=uk_rules['support'], size=8, colorscale='Viridis', opacity=0.8),
    text=f"Support: {uk_rules['support']}, Confidence: {uk_rules['confidence']}, Lift: {uk_rules['lift']}"
)

fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='Support', yaxis_title='Confidence', zaxis_title='Lift'))
fig.update_layout(title='Interactive 3D Scatter Plot of Support, Confidence, and Lift')
fig.show()



# In[26]:


top_rules = uk_rules[
    (uk_rules['lift'] >= 3) &
    (uk_rules['confidence'] >= 0.5) &
    (uk_rules['support'] >= 0.03)
].nlargest(10, 'confidence')

top_rules_pivot = top_rules.pivot(index='antecedents', columns='consequents', values='confidence')

plt.figure(figsize=(14, 12))
sns.heatmap(top_rules_pivot, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Confidence'})
plt.title('Top 10 Association Rules Heatmap based on Confidence')
plt.show()



# <font size="39"><b>GERMANY Basket</b></font>

# <p style="font-size:18px">Germany Basket Generated Rules</p>

# In[27]:


germany_rules.head(100)


# In[28]:


germany_rules.sort_values('confidence', ascending=False)


# In[29]:


germany_rules[(germany_rules['lift'] >= 3) & (germany_rules['confidence'] >= 0.5) & (germany_rules['support'] >= 0.03)]



# <p style="font-size:18px">Germany Basket Predicted Visualization</p>

# In[30]:


top_rules = germany_rules.nlargest(10, 'lift')

rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(12, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['lift'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by Lift in Germany')
plt.show()


# In[31]:


top_rules = germany_rules.nlargest(10, 'confidence')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(14, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['confidence'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by confidence in Germany')
plt.show()


# In[32]:


top_rules = germany_rules.nlargest(10, 'support')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(14, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['support'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.axis('equal')
plt.title('Top 10 Association Rules by support in Germany')
plt.show()


# In[33]:


fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=germany_rules['support'],
    y=germany_rules['confidence'],
    z=germany_rules['lift'],
    mode='markers',
    marker=dict(color=germany_rules['support'], size=8, colorscale='Viridis', opacity=0.8),
    text=f"Support: {germany_rules['support']}, Confidence: {germany_rules['confidence']}, Lift: {germany_rules['lift']}"
)

fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='Support', yaxis_title='Confidence', zaxis_title='Lift'))
fig.update_layout(title='Interactive 3D Scatter Plot of Support, Confidence, and Lift')
fig.show()



# In[34]:


top_rules = germany_rules[
    (germany_rules['lift'] >= 3) &
    (germany_rules['confidence'] >= 0.5) &
    (germany_rules['support'] >= 0.03)
].nlargest(10, 'confidence')

top_rules_pivot = top_rules.pivot(index='antecedents', columns='consequents', values='confidence')

plt.figure(figsize=(14, 12))
sns.heatmap(top_rules_pivot, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'confidence'})
plt.title('Top 10 Association Rules Heatmap based on Confidence')
plt.show()



# <font size="39"><b>FRANCE Basket</b></font>

# <p style="font-size:18px">FRANCE Basket Generated Rules</p>

# In[35]:


france_rules.head(100)


# In[36]:


france_rules[ (france_rules['lift'] >= 3) & (france_rules['confidence'] >= 0.5) & (france_rules['support'] >= 0.03)] 



# In[37]:


france_rules.sort_values('confidence', ascending=False)


# <p style="font-size:18px">France Basket Predicted Visualization</p>

# In[38]:


top_rules = france_rules.nlargest(10, 'lift') 
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(12, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['lift'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 24})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 18})
plt.axis('equal')
plt.title('Top 10 Association Rules by Lift in France')
plt.show()


# In[39]:


top_rules = france_rules.nlargest(10, 'confidence')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(12, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['confidence'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 24})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 18})
plt.axis('equal')
plt.title('Top 10 Association Rules by Confidence in France')
plt.show()


# In[40]:


top_rules = france_rules.nlargest(10, 'support')
rule_labels = [f"{str(antecedent)} -> {str(consequent)}" for antecedent, consequent in zip(top_rules['antecedents'], top_rules['consequents'])]
plt.figure(figsize=(12, 12))
colors = plt.cm.Paired(range(len(top_rules)))
patches, texts, autotexts = plt.pie(top_rules['support'], autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 24})
plt.legend(patches, rule_labels, title='Rules', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 18})
plt.axis('equal')
plt.title('Top 10 Association Rules by Support in France')
plt.show()


# In[41]:


fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=france_rules['support'],
    y=france_rules['confidence'],
    z=france_rules['lift'],
    mode='markers',
    marker=dict(color=france_rules['support'], size=8, colorscale='Viridis', opacity=0.8),
    text=f"Support: {france_rules['support']}, Confidence: {france_rules['confidence']}, Lift: {france_rules['lift']}"
)

fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='Support', yaxis_title='Confidence', zaxis_title='Lift'))
fig.update_layout(title='Interactive 3D Scatter Plot of Support, Confidence, and Lift')
fig.show()


# In[42]:


top_rules = france_rules[
    (france_rules['lift'] >= 3) &
    (france_rules['confidence'] >= 0.5) &
    (france_rules['support'] >= 0.03)
].nlargest(10, 'confidence')

top_rules_pivot = top_rules.pivot(index='antecedents', columns='consequents', values='confidence')

plt.figure(figsize=(14, 12))
sns.heatmap(top_rules_pivot, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'confidence'}, annot_kws={'size': 16})

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Top 10 Association Rules Heatmap based on Confidence', fontsize=24)

plt.show()

