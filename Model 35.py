
# coding: utf-8

# In[1]:


#Packages
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[4]:


#Import data
data = pd.read_csv('/Users/celestinoedano/Desktop/timesheet data/casual.csv', names = ['f1','f2','f3','f4','f5','f6','f7'])


# In[5]:


data


# In[6]:


data["f7"].value_counts()


# In[8]:


sns.FacetGrid(data, hue = "f7", size = 5)     .map(plt.scatter, "f4", "f1")     .add_legend()


# In[9]:


#Map data into arrays
ot15 = np.asarray([1,0,0])
ot20 = np.asarray([0,1,0])
wk = np.asarray([0,0,1])
data['f7'] = data['f7'].map({'OT15': ot15, 'OT20': ot20, 'WK': wk})


# In[10]:


data


# In[11]:


#Data shuffling
data = data.iloc[np.random.permutation(len(data))]


# In[12]:


data


# In[13]:


#Reset the indexing
data = data.reset_index(drop = True)


# In[14]:


data


# In[17]:


#Training set
x_input = data.loc[0:305, ['f1','f2','f3','f4','f5','f6']]
temp = data['f7']
y_input = temp[0:306]

#Testing data
x_test = data.loc[306:405, ['f1','f2','f3','f4','f5','f6']]
y_test = temp[306:406]


# In[18]:


#Placeholders & variables. 6 features as input and 3 awards as output.
x = tf.placeholder(tf.float32, shape = [None, 6])
y_ = tf.placeholder(tf.float32, shape = [None, 3])

#Weight and Biases
W = tf.Variable(tf.zeros([6, 3]))
b = tf.Variable(tf.zeros([3]))


# In[19]:


#Softmax algorithm (model)
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[21]:


#Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))


# In[22]:


#Optimizer
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

#Calculating the model's accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[24]:


#Session parameters
sess = tf.InteractiveSession()

#Initializing variables
init = tf.global_variables_initializer()
sess.run(init)

#Number of iterations
epoch = 2000


# In[25]:


for step in xrange(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
       print c


# In[28]:


#Random testing @ 400
a = data.loc[400, ['f1','f2','f3','f4','f5','f6']]
b = a.values.reshape(1, 6)
largest = sess.run(tf.argmax(y, 1), feed_dict = {x: b})[0]
if largest == 0:
    print "Overtime Rate OT15"
elif largest == 1:
    print "Overtime Rate OT20"
else :
    print "Overtime Rate Weekend Multiplier"


# In[29]:


print sess.run(accuracy, feed_dict = {x: x_test, y_: [t for t in y_test.as_matrix()]})

