# -*- coding: utf-8 -*-
"""
Advanced Analytics Lunch & Learn

Clustering Analysis with KMeans
"""

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# I. Generate Data
#-----------------
c1_n = 100
c2_n = 150
c3_n = 300

c1_color = 'b'
c2_color = 'grey'
c3_color = 'r'

# order frequency
c1_freq = [round(np.random.normal(90, 15)) for _ in range(c1_n)]
c2_freq = [round(np.random.normal(60, 15)) for _ in range(c2_n)]
c3_freq = [round(np.random.normal(30, 10)) for _ in range(c3_n)]
sns.distplot(c1_freq + c2_freq + c3_freq)

# average order amt
c1_amt = [np.random.normal(10000, 2000) for _ in range(c1_n)]
c2_amt = [np.random.normal(4000, 1500) for _ in range(c2_n)]
c3_amt = [np.random.normal(3000, 1000) for _ in range(c3_n)]
sns.distplot(c1_amt + c2_amt + c3_amt)

# number of employees
c1_emp = [round(np.random.normal(700, 200)) for _ in range(c1_n)]
c2_emp = [round(np.random.normal(300, 100)) for _ in range(c2_n)]
c3_emp = [round((np.random.chisquare(2) * 15)) for _ in range(c3_n)]
sns.distplot(c1_emp + c2_emp + c3_emp)

# II. Clean Values
#-----------------
lower_bound_dict = {'freq': 4,
                    'amt': 50,
                    'emp': 3}

def scrub_bad_values(lower_bound_dict, class_freq, class_amt, class_emp):
    drop_indexes = []
    for i in range(len(class_freq)):
        if (class_freq[i] < lower_bound_dict['freq'] or
            class_amt[i] < lower_bound_dict['amt'] or
            class_emp[i] < lower_bound_dict['emp']):
            drop_indexes.append(i)
    drop_indexes.reverse()
    for i in drop_indexes:
        class_freq.pop(i)
        class_amt.pop(i)
        class_emp.pop(i)
    return (class_freq, class_amt, class_emp)
    
c1_freq, c1_amt, c1_emp = scrub_bad_values(lower_bound_dict, c1_freq, c1_amt, c1_emp)
c2_freq, c2_amt, c2_emp = scrub_bad_values(lower_bound_dict, c2_freq, c2_amt, c2_emp)
c3_freq, c3_amt, c3_emp = scrub_bad_values(lower_bound_dict, c3_freq, c3_amt, c3_emp)

# III. Visualize Data
#--------------------
# freq & amt scatter plot
sns.scatterplot(c1_freq, c1_amt)
sns.scatterplot(c2_freq, c2_amt)
sns.scatterplot(c3_freq, c3_amt)

# freq & emp scatter plot
sns.scatterplot(c1_freq, c1_emp)
sns.scatterplot(c2_freq, c2_emp)
sns.scatterplot(c3_freq, c3_emp)

# emp & amt scatter plot
sns.scatterplot(c1_emp, c1_amt)
sns.scatterplot(c2_emp, c2_amt)
sns.scatterplot(c3_emp, c3_amt)

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(c1_freq, c1_amt, c1_emp, c=c1_color)
ax.scatter(c2_freq, c2_amt, c2_emp, c=c2_color)
ax.scatter(c3_freq, c3_amt, c3_emp, c=c3_color)

ax.set_xlabel('Days Since Last Purchase')
ax.set_ylabel('Order Amount')
ax.set_zlabel('Number of Employees')
ax.set(title='Data by Actual Class')

# collate data
freq = np.array(c1_freq + c2_freq + c3_freq)
amt = np.array(c1_amt + c2_amt + c3_amt)
emp = np.array(c1_emp + c2_emp + c3_emp)

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(freq, amt, emp, c='grey')

ax.set_xlabel('Days Since Last Purchase')
ax.set_ylabel('Order Amount')
ax.set_zlabel('Number of Employees')
ax.title('Data Without Classes')

# IV. K-Means
#------------
# 1. prep data
data = np.array([[f, a, e] for f, a, e in zip(freq, amt, emp)])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_2 = scaler.fit_transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], c='grey')
ax.set_xlabel('Days Since Last Purchase')
ax.set_ylabel('Order Amount')
ax.set_zlabel('Number of Employees')
ax.set(title='Scaled Data')

# 2. WCSS graph
wcss = []
for i in range(1, 11):
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data_2)
    wcss.append(kmeans.inertia_)
    
ax = sns.lineplot(range(1, 11), wcss)
ax.set_xlabel('# of Clusters')
ax.set_ylabel('WCSS')
ax.set(title='WCSS "Elbow" Chart')

# 3. Apply K-Means
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
y_means = kmeans.fit_predict(data_2)

# 4. Visualize 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[y_means == 0, 0], data[y_means == 0, 1], data[y_means == 0, 2], c='r')
ax.scatter(data[y_means == 1, 0], data[y_means == 1, 1], data[y_means == 1, 2], c='b')
ax.scatter(data[y_means == 2, 0], data[y_means == 2, 1], data[y_means == 2, 2], c='grey')

ax.set_xlabel('Days Since Last Purchase')
ax.set_ylabel('Order Amount')
ax.set_zlabel('Number of Employees')

# 5. Calculate centroid location
c1_centroid = [data[y_means==0, 0].mean(), data[y_means==0, 1].mean(), data[y_means==0, 2].mean()]
c2_centroid = [data[y_means==1, 0].mean(), data[y_means==1, 1].mean(), data[y_means==1, 2].mean()]
c3_centroid = [data[y_means==2, 0].mean(), data[y_means==2, 1].mean(), data[y_means==2, 2].mean()]
centroids = np.array([c1_centroid, c2_centroid, c3_centroid])

# 6. Visualize 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[y_means == 0, 0], data[y_means == 0, 1], data[y_means == 0, 2], c='r')
ax.scatter(data[y_means == 1, 0], data[y_means == 1, 1], data[y_means == 1, 2], c='b')
ax.scatter(data[y_means == 2, 0], data[y_means == 2, 1], data[y_means == 2, 2], c='grey')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='g', s=500, alpha=1)

ax.set_xlabel('Days Since Last Purchase')
ax.set_ylabel('Order Amount')
ax.set_zlabel('Number of Employees')
plt.legend()

# V. 2D solution
#---------------
# 2d labeled plot - Frequency & Amount
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(freq, amt, c='grey')

ax.set_xlabel('Days Since Last Purchase', fontsize=20)
ax.set_ylabel('Order Amount', fontsize=20)
ax.set_title('Unlabeled Data', fontsize=20)

# 
data = np.array([[f, a] for f, a in zip(freq, amt)])
data_2 = scaler.fit_transform(data)

# 2. WCSS graph
wcss = []
for i in range(1, 11):
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data_2)
    wcss.append(kmeans.inertia_)
    
ax = sns.lineplot(range(1, 11), wcss)
ax.set_xlabel('# of Clusters', fontsize=20)
ax.set_ylabel('WCSS', fontsize=20)
ax.set_title('WCSS "Elbow" Chart', fontsize=20)

# 3. Apply K-Means
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
y_means = kmeans.fit_predict(data_2)

# 4. Visualize 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[y_means == 0, 0], data[y_means == 0, 1], c='r')
ax.scatter(data[y_means == 1, 0], data[y_means == 1, 1], c='b')
ax.scatter(data[y_means == 2, 0], data[y_means == 2, 1], c='g')

ax.set_xlabel('Days Since Last Purchase', fontsize=20)
ax.set_ylabel('Order Amount', fontsize=20)
ax.set_title('Clustered Data', fontsize=25)