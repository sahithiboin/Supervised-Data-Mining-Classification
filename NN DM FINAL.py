
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
iris = datasets.load_iris()
print(iris)


# In[2]:


print(iris.data[0:10,:])


# In[3]:


plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:,0], iris.data[:,3], c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[3])


# In[4]:


print("Targets: " + str(iris.target_names))
print("Features: " + str(iris.feature_names))


# In[5]:


import numpy as np
#np.array([[w_1_1, w_1_2], [w_2_1, w_2_2])
mult_matrix = np.array([[-0.2, 0.2], [-1.0, 1.0]])
bias = np.array([0.1, -0.1])
for features in iris.data:
    our_features = [features[2]-3, features[3]-2]
    print(our_features)
    a = np.matmul(our_features, mult_matrix)
    a = a + bias
    print("activations: " + str(a))


# In[6]:


import numpy as np

def sigmoid(activations):
    return 1 / (1 + np.exp(-activations))

#np.array([[w_1_1, w_1_2], [w_2_1, w_2_2])
mult_matrix = np.array([[-0.2, 0.2], [-1.0, 1.0]])
bias = np.array([0.1, -0.1])

for features in iris.data:
    our_features = [features[2]-3, features[3]-2]
    print(our_features)
    a = np.matmul(our_features, mult_matrix)
    a = a + bias
    a = sigmoid(a)
    print("activations: " + str(a))


# In[7]:


import tensorflow as tf
tf.reset_default_graph()

n_input = len(iris.data[0])
n_output = 3 # [0,1,2]... set(iris.target)

input_shape = [None,n_input]
inputplaceholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name="input_placeholder") # https://www.tensorflow.org/api_docs/python/tf/placeholder

weights = tf.Variable(tf.random_normal([n_input, n_output]), name="weights")
biases = tf.Variable(tf.zeros([n_output]), name="biases")

print(weights)
print(biases)

layer_1 = tf.matmul(inputplaceholder, weights)
layer_2 = tf.add(layer_1, biases)
outputlayer = tf.nn.sigmoid(layer_2)


# In[8]:


learning_rate = 0.1

labelsplaceholder = tf.placeholder(dtype=tf.float32, shape=[None,n_output], name="labels_placeholder")
cost = tf.losses.mean_squared_error(labelsplaceholder, outputlayer) # https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error

print(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) #https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer


# In[9]:


init = tf.global_variables_initializer() # https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
sess = tf.Session() # https://www.tensorflow.org/api_docs/python/tf/Session
sess.run(init)


# In[10]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(iris.data)
scaled_data = scaler.transform(iris.data)


# In[11]:


import random

mydata = list(zip(scaled_data, iris.target))

# for x in mydata:
#     print(x)

batch_size = 10
iterations = 400

history_loss = list()
for _ in range(iterations):
    inputdata = list()
    output_data = list()
    for _ in range(batch_size):
        input_output_pairs = random.choice(mydata)
        inputdata.append(input_output_pairs[0])
        output_one_hot = [0.0,0.0,0.0]
        output_one_hot[input_output_pairs[1]] = 1.0
        output_data.append(output_one_hot)

    res_optimizer, res_cost = sess.run([optimizer, cost], feed_dict={inputplaceholder: inputdata, labelsplaceholder: output_data})
    print(res_cost)
    history_loss.append(res_cost)


# In[12]:


plt.plot(history_loss)


# In[13]:


correct_predictions = 0
for i in range(len(scaled_data)):
    predicted_by_network = sess.run(outputlayer, feed_dict={inputplaceholder: [scaled_data[i]]})
    print("input: %s, expected: %s, predicted: %s " % (str(scaled_data[i]), str(iris.target[i]), str(predicted_by_network)))
    if np.argmax(predicted_by_network) == iris.target[i]:
        correct_predictions += 1

print("Correct_predictions: " + str(correct_predictions) + "/" + str(len(scaled_data)) + " Accuracy: " + str(correct_predictions/len(scaled_data)))


# In[14]:


from sklearn import model_selection 

x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

## scaling - see next chapter
scaler = preprocessing.StandardScaler().fit(x_train)
scaled_data_train = scaler.transform(x_train)
scaled_data_test = scaler.transform(x_test)


# In[15]:


init = tf.global_variables_initializer() # https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer
sess = tf.Session() # https://www.tensorflow.org/api_docs/python/tf/Session
sess.run(init)

mydata = list(zip(scaled_data_train, y_train))


batch_size = 10

history_loss = list()
for _ in range(400):
    inputdata = list()
    outputlogits = list()
    for _ in range(batch_size):
        input_output_pairs = random.choice(mydata)
        inputdata.append(input_output_pairs[0])
        output_expected = [0.0,0.0,0.0]
        output_expected[input_output_pairs[1]] = 1.0
        outputlogits.append(output_expected)

    res_optimizer, res_cost = sess.run([optimizer, cost], feed_dict={inputplaceholder: inputdata, labelsplaceholder: outputlogits})
    print(res_cost)
    history_loss.append(res_cost)


# In[16]:


plt.plot(history_loss)


# In[17]:


logit_y_test = list()
for label in y_test: 
    toadd = [0.0,0.0,0.0]
    toadd[label] = 1.0
    logit_y_test.append(toadd)
res_cost, predicted = sess.run([cost, outputlayer], feed_dict={inputplaceholder: scaled_data_test, labelsplaceholder: logit_y_test})


# In[18]:


print(res_cost)


# In[19]:


correct_predictions = 0
for index in range(len(y_test)):
    print("Label: %d, predicted: %s" % (y_test[index], predicted[index]))
    if y_test[index] == np.argmax(predicted[index]):
        correct_predictions += 1
print(correct_predictions)
print(len(y_test))


# In[20]:


X = iris.data
y = iris.target
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]


# In[21]:


# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# In[22]:


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)


# In[23]:


# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# In[24]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[25]:


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[26]:


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[27]:


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


# In[28]:


# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])


# In[29]:


# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# In[30]:


# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

