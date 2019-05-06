
# Performing Principal Component Analysis (PCA) - Lab

## Introduction

Now that you have high level overview of PCA as well as some of the details in the algorithm itself, its time to practice implementing PCA on your own using the NumPy package. 

## Objectives

You will be able to:
    
* Implement PCA from scratch using NumPy

## Import the Data

To start, import the data stored in the file 'foodusa.csv'.


```python
#Your code here
```

## Normalize the Data

Next, normalize your data by subtracting the feature mean from each of the columns


```python
#Your code here
```

## Calculate the Covariance Matrix

The next step for PCA is to calculate to covariance matrix for your normalized data. Do so here.


```python
#Your code here
```

## Calculate the Eigenvectors

Next, calculate the eigenvectors for your covariance matrix.


```python
#Your code here
```

## Sorting the Eigenvectors to Determine Primary Components

Great! Now that you have the eigenvectors and their associated eigenvalues, sort the eigenvectors based on their eigenvalues!


```python
#Your code here
```

## Reprojecting the Data

Finally, reproject the dataset using your eigenvectors. Reproject the dataset down to 2 dimensions.


```python
#Your code here
```

## Summary

Well done! You've now coded PCA on your own using NumPy! With that, it's time to look at further application of PCA.
