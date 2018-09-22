Title: Labeling for Supervised Learning in Finance
Slug: finance_labeling
Date: 2018-06-22 12:00
Category: Mahine Learning
Tags: Finance
Author: Tomoaki Fujii
Status: published

Predicting future stock price movement is known difficult due to low signal-to-noise ratio
and non stationary price distribution.
Off-the-shelf successful ML algorithms often end up giving you disappointed results.
Indeed, a lot of ML quant hedge funds show up and disappear every year. 

![southpark_finance](https://media.giphy.com/media/3o6ZtiGv3tTlXTsWli/giphy.gif)

To deal with this difficult problems, we need to set up ML formalizations more carefully.
In this article, we focus on considering how to label data and execute supervised learning.
Most approaches here are based on a recently published book, `Advances in Financial Machine Learning`[[1]](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089).
I highly recommend you to check it for various useful methods.



#### Spoiler Alert!
The prediction result itself is not impressive.
We just go through the basic way to formalize the problem.
I will write about how to validate model and tune hyperparameters to improve the performance
in a different blog post. 


# Classification Approach

Predicting the direction of stock prices are used to make signals for algorithm trading.
This task is formalized as classification, where classifiers predict if the future price goes up or down.
Then, the resultant outputs of classifiers can be fed into trading algorithms as signals,
e.g., buying(selling) a stock when the prediction is positive(negative).
To treat the task as classification, we need to get labels according to the future price.
One of the approaches I have seen a lot is simply labeling with next date price direction.
Labels getting through this approach may be contaminated by noise due to low signal-to-noise ratio,
i.e., strong noise compared to signal.

For example, even if the next date return distribution has positive mean,
the price may end up going downward due to the noise.
Let's see this from more mathematical perspective. 

Let $P_n$ and $P_0$ be n date forward and current stock price.
They are related through the following equation:

$$P_n = P_0 \prod_{i=1}^N (1 + r_i)$$
, where $r_i$ is return for each date.

If $r_i << 1$, $\forall i$, we approximate this relation by
$$P_n \sim P_0 (1 + \sum_{i=1}^n r_i)$$

If all returns are sampled according to the same normal distribution distribution,
i.e., $r_i \sim N(\mu, \sigma), \forall i$,
$$\sum_{i=1}^N r_i \sim N(n\mu, \sqrt{n} \sigma)$$.

Thus, mean value grows faster than standard deviation.
This result implies that labeling price with further future gives you more reliable labels.
For example, consider the case where mean is positive. We want to label the direction positive up to one standard deviation.
In order to achieve this, we need to take n such that

$$n \mu \geq \sqrt{n} \sigma$$
$$ n \geq (\frac{\sigma}{\mu})^2$$

If we use n larger than the square of noise ratio,
the price direction will be labeled correctly with about 84 percent.

## Synthetic Data
Let's see the above statement of above discussion in a synthetic data.

```python
import numpy as np

mu = 0.1
sig = 1.
N = 400
n_samples = 1000
x = np.arange(N)
samples = []
for i in range(n_samples):
    rs = np.random.normal(mu, sig, N)
    rs = np.cumsum(rs)
    samples.append(rs)
samples = np.array(samples)
r_mu = np.mean(samples, axis=0)
r_sig = np.std(samples, axis=0)
```

We consider the example where mean is 0.1 and standard deviation is 1.0.
Since the signal-to-noise ratio is 10, we can deduce that 100 steps are required to get reliable labels.

![synthetic]({filename}/images/finance_labeling/synthetic.png)

As we expected, points within one standard deviation resides in positive area after 100 steps.
Before 100 steps, the points are highly likely to end up sitting on negative area.
If you label data point with short term forward points, you may get wrong labels.

##  Labeling
If the above statement is applicable to actual stock price data, let's label data with 10 years forward price! 
Does this make sense? The answer is probably no.
One of the main differences between actual stock price and the synthetic data is stationarity of the distribution.
The price distribution is always changing along with the market condition
 How much forward price you use would be trade-off between reliability and consistency.
 This length has to be tuned as a hyperparameter of your algorithms.
 Although we do not discuss here how to validate model performance correctly and tune hyperparameters,
 we will go through the way to implement a strategy of labeling stock price data. 

As an example, we play with `MicroSoft` daily stock price data.

![MSFT]({filename}/images/finance_labeling/MSFT.png)

In `Advances in Financial Machine Learning`[[1]](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089),
the author suggests `triple barrier method`.
He labels data with the two horizontal and the one vertical barriers.
The horizontal barriers define what price level would be classified as positive or negative while the vertical barrier set how long you look further at maximum for labeling.
More precisely, each data point is labeled by the first barrier hit by the future price.

The implementation is based on Chapter 3 of `Advances in Financial Machine Learning`.
Some implementations are omitted for the sake of simplicity.
Please check out my repository[[2]](https://github.com/jjakimoto/finance_ml) for the full implementation.

First, we set up the vertical barrier.

```python
def get_t1(close, timestamps, num_days):
    """Return horizontal timestamps

    Note
    ----
    Not include the case to hit the vertical line at the end of close.index

    Parameters
    ----------
    close: pd.Series
    timestamps: pd.DatetimeIndex
    num_days: int
        The number of forward dates for vertical barrier

    Returns
    -------
    pd.Series: Vertical barrier timestamps
    """
    t1 = close.index.searchsorted(timestamps + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=timestamps[:t1.shape[0]])
    return t1
    
# We use all data points in this article
timestamps = close.index
num_days = 10
t1 = get_t1(close, timestamps, num_days)
print(t1.head())
```
Output:
```
Date
2000-01-03   2000-01-13
2000-01-04   2000-01-14
2000-01-05   2000-01-18
2000-01-06   2000-01-18
2000-01-07   2000-01-18
Name: Date, dtype: datetime64[ns]
```

Each element defines what timestamp is defined as the vertical barrier. 

To define the horizontal barriers, we need two parameters.
One of them is `trgt`, which defines the scale of barrier width.
The basic idea is that we need to change the width of barriers depending on market conditions.
For example, if we are facing a volatile market, we need to use a wide width.
You can use daily volatilities to set `trgt`.
The daily volatilities are estimated through exponential moving average.

The other parameter is `sltp`:stop loss and take profit.
These parameters give you flexibility to define the width of barriers depending on your preference.
The positive(negative) label barrier is defined by `sltp[0]`(`sltp[1]`) times `trgt`.

```python
def get_touch_idx(close, events, sltp, molecule=None):
    """Return timestamps of when data points touch the barriers

    Parameters
    ----------
    close: pd.Series
        Close price series
    events: pd.DataFrame with columns: 't1', 'trgt', and 'side'
        t1: time stamp of vertical barrier, could be np.nan
        trgt: unit of width of horizontal barriers
        side: Side label for metalabeling
    sltp: list
        Coefficients of width of Stop Loss and Take Profit.
        sltp[0] and sltp[1] correspond to width of stop loss
        and take profit, respectively. If 0 or negative, the barrier
        is siwthced off.
    molecule: list, optional
        Subset of indices of events to be processed

    Returns
    -------
    pd.DataFrame: each colum corresponds to the time to touch the barrier
    """
    # Sample a subset with specific indices
    if molecule is not None:
        _events = events.loc[molecule]
    else:
        _events = events
    touch_idx = pd.DataFrame(index=_events.index)
    # Set Stop Loss and Take Profoit
    if sltp[0] > 0:
        sls = -sltp[0] * _events["trgt"]
    else:
        # Switch off stop loss
        sls = pd.Series(index=_events.index)
    if sltp[1] > 0:
        tps = sltp[1] * _events["trgt"]
    else:
        # Switch off profit taking
        tps = pd.Series(index=_events.index)
    # Replace undefined value with the last time index
    vertical_lines = _events["t1"].fillna(close.index[-1])
    for loc, t1 in vertical_lines.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side
        df = (df / close[loc] - 1) * _events.at[loc, 'side']
        touch_idx.at[loc, 'sl'] = df[df < sls[loc]].index.min()
        touch_idx.at[loc, 'tp'] = df[df > tps[loc]].index.min()
    touch_idx['t1'] = _events['t1'].copy(deep=True)
    return touch_idx
```

`get_touch_idx` gets when and what kind of barriers the future price hits. 

```python
import pandas as pd
from finance_ml.multiprocessing import mp_pandas_obj


def get_events(close, timestamps, sltp, trgt, min_ret=0,
               num_threads=1, t1=None, side=None):
    """Return DataFrame containing infomation defining barriers

    Parameters
    ----------
    close: pd.Series
        Close price series
    timestamps: pd.DatetimeIndex
        sampled points to analyze
    sltp: list
        Coefficients of width of Stop Loss and Take Profit.
        sltp[0] and sltp[1] correspond to width of stop loss
        and take profit, respectively. If 0 or negative, the barrier
        is siwthced off.
    trgt: pd.Series
        Time series of threashold
    min_ret: float, (default 0)
        Minimum value of points to label
    num_threads: int, (default 1)
        The number of threads to use
    t1: pd.Series, optional
        Vertical lines
    side: pd.Series, optional
        Side of trading positions

    Returns
    -------
    pd.DataFrame with columns: 't1', 'trgt', 'type', and 'side'
    """
    # Get sampled target values
    trgt = trgt.loc[timestamps]
    trgt = trgt[trgt > min_ret]
    if len(trgt) == 0:
        return pd.DataFrame(columns=['t1', 'trgt', 'side'])
    # Get time boundary t1
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=timestamps)
    # slpt has to be either of integer, list or tuple
    if isinstance(sltp, list) or isinstance(sltp, tuple):
        _sltp = sltp[:2]
    else:
        _sltp = [sltp, sltp]
    # Define the side
    if side is None:
        # Default is LONG
        _side = pd.Series(1, index=trgt.index)
    else:
        _side = side.loc[trgt.index]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = mp_pandas_obj(func=get_touch_idx,
                             pd_obj=('molecule', events.index),
                             num_threads=num_threads,
                             close=close, events=events, sltp=_sltp)
    # Skip when all of barrier are not touched
    time_idx = time_idx.dropna(how='all')
    events['type'] = time_idx.idxmin(axis=1)
    events['t1'] = time_idx.min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events
```

```python
from finance_ml.stats import get_daily_vol

vol = get_daily_vol(close)
print('volatility')
print(vol.head())

events = get_events(close, timestamps, [2, 2], vol, min_ret=0,
                    num_threads=16, t1=t1, side=None)
print('events')
print(events.head())
```
Output:
```
volatility
Date
2000-01-04         NaN
2000-01-05    0.031374
2000-01-06    0.025522
2000-01-10    0.024588
2000-01-11    0.022054
Name: Close, dtype: float64

events
                   t1      trgt type
Date                                
2000-01-05 2000-01-12  0.031374   sl
2000-01-06 2000-01-18  0.025522   t1
2000-01-10 2000-01-12  0.024588   sl
2000-01-11 2000-01-18  0.022054   tp
2000-01-12 2000-01-14  0.020946   tp
```

`get_events` uses `get_torch_idx` internally and obtains labels.

Output, `events`, contains the followings:
- `t1`, when the barrier is touched
- `trgt`, scale used to define horizontal barriers
- `type`, which barrier is touched


Next, we define `get_sizes`, which generates numerical labels using `events`.
When labeling points hitting the vertical barrier, there are two possible choices.
One of them is assigning the sign of the return at the hitting point.
The other way is using another label for hitting the vertical barrier.


In this article, we take the former approach to get binary labels.


```python
def get_sizes(close, events, sign_label=True):
    """Return bet sizes

    Parameters
    ----------
    close: pd.Series
    events: pd.DataFrame
        t1: time of barrier
        type: type of barrier - tp, sl, or t1
        trgt: horizontal barrier width
        side: position side
    sign_label: bool, (default True)
        If True, assign label for points touching vertical
        line accroing to return's sign

    Returns
    -------
    pd.Series: bet sizes
    """
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    # All used indices
    time_idx = events.index.union(events['t1'].values).drop_duplicates()
    close = close.reindex(time_idx, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = close.loc[events['t1'].values].values / close.loc[
        events.index].values - 1.
    if 'side' in events:
        out['ret'] *= events['side']
        out['side'] = events['side']
    out['size'] = np.sign(out['ret'])
    if sign_label:
        out['size'] = np.sign(out['ret'])
        out.loc[out['ret'] == 0, 'size'] = 1.
    else:
        # 0 when touching vertical line
        out['size'].loc[events['type'] == 't1'] = 0
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'size'] = 0
    return out
    
labels = get_sizes(close, events, sign_label=True)
print(labels.head())
```
Output:
```python
                 ret  size
Date                      
2000-01-05 -0.070293  -1.0
2000-01-06  0.048273   1.0
2000-01-10 -0.057372  -1.0
2000-01-11  0.054311   1.0
2000-01-12  0.060864   1.0
```

## Prediction

Finally, we come to the stage for prediction. Yay! In this stage, we simply test results through split datasets: training and test.

We simply use trailing histories of volume and close for input features.


![MSFT_Volume]({filename}/images/finance_labeling/MSFT_Volume.png)

We split data into from `2000-01-01` to `2017-08-31` for training and from `2017-09-01` to `2018-03-31` for test.
```python
# Separate data time stamps
def get_partial_index(df, start=None, end=None):
    if start is not None:
        df = df.loc[df.index >= start]
    if end is not None:
        df = df.loc[df.index <= end]
    return df.index

train_end = '2017-08-31'
test_start = '2017-09-01'
train_idx = get_partial_index(df, end=train_end)
test_idx = get_partial_index(df, start=test_start)


def generate_features(close, volume, label, timestamps, timelag):
    index = close.index
    data = []
    for i in range(1, timelag):
        # Normalize
        data.append(close.shift(i).values / close.values)
        data.append(volume.shift(i).values / volume.values)

    features = pd.DataFrame(np.stack(data, axis=1), index=index)
    features = features.loc[timestamps].dropna()
    label = label.dropna()
    time_idx = features.index & label.index
    y = label.loc[time_idx].values
    label_map = {-1: 0, 1: 1}
    y = np.array([label_map[y_i] for y_i in y]).astype(int)
    X = features.loc[time_idx].values
    return X, y


timelag = 30
train_X, train_y = generate_features(close, volume, labels['size'], train_idx, timelag=timelag)
test_X, test_y = generate_features(close, volume, labels['size'], test_idx, timelag=timelag)
```

Note that close and volume features are normalized with the current value [[3]](https://arxiv.org/pdf/1706.10059.pdf).
Intuitively, the scales of close and volume themselves do not have any meanings. The value in comparison to the current close and value are rather essential information. This normalization allows you to build models irrelevant to the scales.


We build up Neural Net classifier with `PyTorch` and my utility repository
`torch_utils`[[4]](https://github.com/jjakimoto/PyTorch-Utils).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import torch.optim as optim
from sklearn.metrics import accuracy_score

from torch_utils.datasets import NumpyDataset
from torch_utils.training import train_step, test_step

input_dim = train_X.shape[1]
output_dim = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def predict(self, x, threshold=.5):
        x = self.forward(x)
        x = F.sigmoid(x)
        return x > threshold

batch_size = 32
train_loader = tdata.DataLoader(NumpyDataset(train_X, train_y[:, None].astype(float)),
                                             batch_size=batch_size, shuffle=True)
test_loader = tdata.DataLoader(NumpyDataset(test_X, test_y[:, None].astype(float)),
                                            batch_size=batch_size)
                                            

n_epochs = 1000
model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_func = F.binary_cross_entropy_with_logits
score_func = accuracy_score
for i in range(n_epochs):
    train_step(model, train_loader, optimizer,
               loss_func=loss_func, score_func=score_func,
               epoch=i, log_interval=0, silent=True)
    if i % 100 == 0:
        test_step(model, test_loader, loss_func=loss_func, score_func=score_func)

model.eval()
output = model.predict(torch.tensor(test_X).float())
accuracy = accuracy_score(test_y, output)
print(f'Test Accuracy: {accuracy:.4g}')
```
Output:
```python
Test Accuracy: 0.5229
```

TADA! The resultant test accuracy is `0.5229`.....
Ummmmmm......., this is no better than chance.
One of the possible reasons for this is overfitting.
We need to tweak the model architecture and way to train models.
We can also consider that classification might be difficult for finance.
Here is the explanation in the Quora [[5]](https://www.quora.com/Loss-cross-entropy-is-decreasing-but-accuracy-remains-the-same-while-training-convolutional-neural-networks-How-can-it-happen).
Even if a model is able to learn the distribution, it might be difficult to predict the correct label under the noisy situation.


# Regression Approach

As we see previously, classification is difficult. It might make more sense to predict the price or return itself.
We try to predict future returns based on the same input features as classification.
We need to consider how many days future to look forward to define target returns.

To compare performances among different `num_day` parameters: `1, 2, 3, 4, 5, 10, 20`,
we use free scale metrics `np.mean(np.abs(y_pred - y)) / np.std(y)`.
Here are the results. 

![regression_1days]({filename}/images/finance_labeling/regression_1days.png)
![regression_3days]({filename}/images/finance_labeling/regression_3days.png)
![regression_5days]({filename}/images/finance_labeling/regression_5days.png)
![regression_10days]({filename}/images/finance_labeling/regression_10days.png)
![regression_20days]({filename}/images/finance_labeling/regression_20days.png)

![error_regression]({filename}/images/finance_labeling/error_regression.png)

We do not see any specific relation between the length of days forward and performance.
To find more reliable results, we need to validate model and tune hyperparameters.

# Wrap Up

The results we have seen in this article looks disappointed. We can mainly consider the following reasons:
1. The small number of data points
2. Require to choose appropriate models

The first reason comes from the fact that the only single path is given and exactly the same pattern will never show up. Generally speaking, ML algorithms needs to see a lot of samples from the same distribution. Especially, when using Neural Network, a lot of data points are required.

For the second reason, I did not spend a lot of times on model selection.
We need to set up proper way to validate model performances, which will be discussed in a future blog post.
As one of the research direction, we can consider Bayesian approaches.
Due to the noisy nature of financial data,
Bayesian approaches help you avoid overfitting and give you more appropriate confidence levels of predictions. 

# References
* [1] [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
* [2] [finance_ml](https://github.com/jjakimoto/finance_ml)
* [3] [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059.pdf)
* [4] [PyTorch-Utils](https://github.com/jjakimoto/PyTorch-Utils)
* [5] [Loss (cross entropy) is decreasing but accuracy remains the same while training convolutional neural networks. How can it happen?](https://www.quora.com/Loss-cross-entropy-is-decreasing-but-accuracy-remains-the-same-while-training-convolutional-neural-networks-How-can-it-happen)