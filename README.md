# Performing Principal Component Analysis (PCA) - Lab

## Introduction

Now that you have a high-level overview of PCA, as well as some of the details of the algorithm itself, it's time to practice implementing PCA on your own using the NumPy package. 

## Objectives

You will be able to:
    
* Implement PCA from scratch using NumPy

## Import the data

- Import the data stored in the file `'foodusa.csv'` (set `index_col=0`)
- Print the first five rows of the DataFrame 


```python
import pandas as pd
data = None


```

## Normalize the data

Next, normalize your data by subtracting the mean from each of the columns.


```python
data = None
data.head()
```

## Calculate the covariance matrix

The next step is to calculate the covariance matrix for your normalized data. 


```python
cov_mat = None
cov_mat
```

## Calculate the eigenvectors

Next, calculate the eigenvectors and eigenvalues for your covariance matrix. 


```python
import numpy as np
eig_values, eig_vectors = None
```

## Sort the eigenvectors 

Great! Now that you have the eigenvectors and their associated eigenvalues, sort the eigenvectors based on their eigenvalues to determine primary components!


```python
# Get the index values of the sorted eigenvalues
e_indices = None

# Sort 
eigenvectors_sorted = None
eigenvectors_sorted
```

## Reprojecting the data

Finally, reproject the dataset using your eigenvectors. Reproject this dataset down to 2 dimensions.


```python

```

## Summary

Well done! You've now coded PCA on your own using NumPy! With that, it's time to look at further applications of PCA.
