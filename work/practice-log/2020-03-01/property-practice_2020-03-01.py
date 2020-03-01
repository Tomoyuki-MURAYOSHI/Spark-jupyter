#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import dask.array as dd


# - @property test

# In[2]:


class Resistor(object):
    def __init__(self, ohms):
        self.ohms = ohms
        self.voltage = 0
        self.current = 0


# In[3]:


r1 = Resistor(50e3)


# In[4]:


r1.ohms


# In[ ]:





# In[5]:


class VoltageResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)
        self._voltage = 0
    
    @property
    def voltage(self):
        return self._voltage
    
    @voltage.setter
    def voltage(self, voltage):
        self._voltage = voltage
        self.current = self._voltage / self.ohms
    


# In[ ]:





# In[18]:


class BoundedResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)
    
    @property
    def ohms(self):
        return self._ohms
    
    @ohms.setter
    def ohms(self, ohms):
        if ohms <= 0:
            raise ValueError(f"{ohms} ohms must be > 0")
        self._ohms = ohms


# In[19]:


r3 = BoundedResistance(1e3)


# In[20]:


r3.ohms = 0


# In[21]:


BoundedResistance(-3)


# In[ ]:





# In[ ]:





# In[28]:


from datetime import timedelta
from datetime import datetime


# In[35]:


class Bucket:
    def __init__(self, period):
        self.period_delta = timedelta(seconds=period)
        self.reset_time = datetime.now()
        self.quota = 0

    def __repr__(self):
        return f"Bucket(quota={self.quota})"


def fill(bucket, amount):
    now = datetime.now()
    if now - bucket.reset_time > bucket.period_delta:
        bucket.quota = 0
        bucket.reset_time = now
    bucket.quota += amount
    

def deduct(bucket, amount):
    now = datetime.now()
    if now - bucket.reset_time > bucket.period_delta:
        return False
    if bucket.quota - amount < 0:
        return False
    bucket.quota -= amount
    return True


# In[36]:


bucket = Bucket(60)


# In[37]:


fill(bucket, 100)


# In[38]:


print(bucket)


# In[ ]:





# In[ ]:




