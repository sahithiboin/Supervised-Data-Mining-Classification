
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import interp

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


# In[2]:


data=pd.read_csv('/Users/my/bezdekIris.csv', names=['f1','f2','f3','f4','f5'])


# In[3]:


data


# In[4]:


data["f5"].value_counts()


# In[5]:


sns.FacetGrid(data, hue="f5", size=5)    .map(plt.scatter, "f1", "f2")    .add_legend()


# In[6]:


#map data into arrays
s=np.asarray([1,0,0])
ve=np.asarray([0,1,0])
vi=np.asarray([0,0,1])
data['f5'] = data['f5'].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})


# In[7]:


data


# In[8]:


#shuffle the data
data=data.iloc[np.random.permutation(len(data))]


# In[9]:


data


# In[10]:


data=data.reset_index(drop=True)


# In[11]:


data


# In[12]:


#training data
x_input=data.ix[0:105,['f1','f2','f3','f4']]
temp=data['f5']
y_input=temp[0:106]
#test data
x_test=data.ix[106:149,['f1','f2','f3','f4']]
y_test=temp[106:150]


# In[13]:


#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])
#weight and bias
W=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))


# In[14]:


# model 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[15]:


#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[16]:


#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[17]:


#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.initialize_all_variables()
sess.run(init)
#number of interations
epoch=2000


# In[18]:


for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
       print(c)


# In[19]:


#random testing at Sn.130
a=data.ix[130,['f1','f2','f3','f4']]
b=a.reshape(1,4)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print("flower is :Iris-setosa")
elif largest==1:
    print("flower is :Iris-versicolor")
else :
    print("flower is :Iris-virginica")


# In[20]:


print(sess.run(accuracy,feed_dict={x: x_test, y_:[t for t in y_test.as_matrix()]}))


# In[22]:


# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape


# In[23]:


# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# In[24]:


# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

