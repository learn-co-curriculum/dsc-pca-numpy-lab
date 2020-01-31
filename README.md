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
data = pd.read_csv('foodusa.csv', index_col=0)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>24.5</td>
      <td>94.5</td>
      <td>73.9</td>
      <td>80.1</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>26.5</td>
      <td>91.0</td>
      <td>67.5</td>
      <td>74.6</td>
      <td>53.3</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>29.7</td>
      <td>100.8</td>
      <td>61.4</td>
      <td>104.0</td>
      <td>59.6</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>22.8</td>
      <td>86.6</td>
      <td>65.3</td>
      <td>118.4</td>
      <td>51.2</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>26.7</td>
      <td>86.7</td>
      <td>62.7</td>
      <td>105.9</td>
      <td>51.2</td>
    </tr>
  </tbody>
</table>
</div>



## Normalize the data

Next, normalize your data by subtracting the mean from each of the columns.


```python
data = data - data.mean()
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>City</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ATLANTA</th>
      <td>-0.791304</td>
      <td>2.643478</td>
      <td>11.604348</td>
      <td>-22.891304</td>
      <td>-7.165217</td>
    </tr>
    <tr>
      <th>BALTIMORE</th>
      <td>1.208696</td>
      <td>-0.856522</td>
      <td>5.204348</td>
      <td>-28.391304</td>
      <td>4.534783</td>
    </tr>
    <tr>
      <th>BOSTON</th>
      <td>4.408696</td>
      <td>8.943478</td>
      <td>-0.895652</td>
      <td>1.008696</td>
      <td>10.834783</td>
    </tr>
    <tr>
      <th>BUFFALO</th>
      <td>-2.491304</td>
      <td>-5.256522</td>
      <td>3.004348</td>
      <td>15.408696</td>
      <td>2.434783</td>
    </tr>
    <tr>
      <th>CHICAGO</th>
      <td>1.408696</td>
      <td>-5.156522</td>
      <td>0.404348</td>
      <td>2.908696</td>
      <td>2.434783</td>
    </tr>
  </tbody>
</table>
</div>



## Calculate the covariance matrix

The next step is to calculate the covariance matrix for your normalized data. 


```python
cov_mat = data.cov()
cov_mat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bread</th>
      <th>Burger</th>
      <th>Milk</th>
      <th>Oranges</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bread</th>
      <td>6.284466</td>
      <td>12.910968</td>
      <td>5.719051</td>
      <td>1.310375</td>
      <td>7.285138</td>
    </tr>
    <tr>
      <th>Burger</th>
      <td>12.910968</td>
      <td>57.077115</td>
      <td>17.507530</td>
      <td>22.691877</td>
      <td>36.294783</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>5.719051</td>
      <td>17.507530</td>
      <td>48.305889</td>
      <td>-0.275040</td>
      <td>13.443478</td>
    </tr>
    <tr>
      <th>Oranges</th>
      <td>1.310375</td>
      <td>22.691877</td>
      <td>-0.275040</td>
      <td>202.756285</td>
      <td>38.762411</td>
    </tr>
    <tr>
      <th>Tomatoes</th>
      <td>7.285138</td>
      <td>36.294783</td>
      <td>13.443478</td>
      <td>38.762411</td>
      <td>57.800553</td>
    </tr>
  </tbody>
</table>
</div>



## Calculate the eigenvectors

Next, calculate the eigenvectors and eigenvalues for your covariance matrix. 


```python
import numpy as np
eig_values, eig_vectors = np.linalg.eig(cov_mat)
```

## Sort the eigenvectors 

Great! Now that you have the eigenvectors and their associated eigenvalues, sort the eigenvectors based on their eigenvalues to determine primary components!


```python
# Get the index values of the sorted eigenvalues
e_indices = np.argsort(eig_values)[::-1] 

# Sort
eigenvectors_sorted = eig_vectors[:, e_indices]
eigenvectors_sorted
```




    array([[-0.02848905, -0.16532108,  0.02135748, -0.18972574, -0.96716354],
           [-0.2001224 , -0.63218494,  0.25420475, -0.65862454,  0.24877074],
           [-0.0416723 , -0.44215032, -0.88874949,  0.10765906,  0.03606094],
           [-0.93885906,  0.31435473, -0.12135003, -0.06904699, -0.01521357],
           [-0.27558389, -0.52791603,  0.36100184,  0.71684022, -0.03429221]])



## Reprojecting the data

Finally, reproject the dataset using your eigenvectors. Reproject this dataset down to 2 dimensions.


```python
eigenvectors_sorted[:2]
```




    array([[-0.02848905, -0.16532108,  0.02135748, -0.18972574, -0.96716354],
           [-0.2001224 , -0.63218494,  0.25420475, -0.65862454,  0.24877074]])



## Summary

Well done! You've now coded PCA on your own using NumPy! With that, it's time to look at further applications of PCA.
