#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install findspark


# In[3]:


pip install pyspark


# In[4]:


import pyspark
import findspark


# In[5]:


findspark.init()
findspark.find()


# In[6]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FP-Growth").getOrCreate()
#spark = SparkSession.builder \
 #   .appName("FPGrowthExample") \
  #  .config("spark.executor.memory", "16g") \
  #  .config("spark.driver.memory", "12g") \
   # .config("spark.executor.cores", "4") \
   # .config("spark.driver.cores", "4") \
    #.getOrCreate()


# In[7]:


from pyspark.sql import SparkSession
#from pyspark.sql.function import collect_list
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType


# In[8]:


df = spark.read.csv("C:\\Users\\Akbar\\Desktop\\data.csv", header=True, inferSchema=True)
df.show()


# In[9]:


df = df.dropna()


# In[10]:


df = df.fillna(0)


# In[ ]:


df= df.dropDuplicates()


# In[ ]:


df= df.filter(df["Quantity"] >= 0)


# In[ ]:


columns_to_drop = ["Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
df = df.drop(*columns_to_drop)


# In[ ]:


from pyspark.sql.functions import lower, col

df_cleaned = df.withColumn("Description", lower(col("Description")))


# In[14]:


transactions = df.groupBy("Country", "InvoiceNo").agg(F.collect_set("StockCode").alias("items"))


# In[15]:


countries = transactions.select("Country").distinct().collect()
for country in countries:
    country_name = country["Country"]
    print(f"Country: {country_name}")


# In[16]:


country_transactions = transactions.filter(F.col("Country") == country_name)


# In[17]:


fpGrowth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.1)


# In[18]:


model = fpGrowth.fit(country_transactions)


# In[19]:


print("Frequent Itemsets: ")
model.freqItemsets.show()


# In[ ]:


print("Association Rules: ")
model.associationRules.show()


# In[16]:


top_items = df.groupBy("StockCode").count().orderBy(F.desc("count")).limit(10)
print("Top 10 Items:")
top_items.show()


# In[17]:


top_countries = transactions.groupBy("Country").count().orderBy(F.desc("count")).limit(10)
print("Top 10 Countries:")
top_countries.show()


# In[18]:


countries = top_countries.select("Country").collect()


# In[ ]:


print("Top 10 Frequent Itemsets:")
model.freqItemsets.orderBy(F.desc("freq")).limit(10).show()


# In[ ]:


print("Top 10 Association Rules:")
model.associationRules.orderBy(F.desc("confidence")).limit(10).show()


# In[ ]:


print("Top 10 Association Rules by Lift:")
model.associationRules.orderBy(F.desc("lift")).limit(10).show()


# In[ ]:


print("Top 10 Association Rules by Support:")
model.associationRules.orderBy(F.desc("support")).limit(10).show()


# In[ ]:


print("Top 10 Association Rules by Confidence:")
model.associationRules.orderBy(F.desc("confidence")).limit(10).show()


# In[ ]:




