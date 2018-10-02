Title: Thought and Trial on Reinforcement Learning for Finance
Slug: rl_portfolio
Date: 2018-10-11 12:00
Category: Machine Learning
Tags: Reinforcement Learning
Author: Tomoaki Fujii
Status: draft


RL(Reinforcement Learning) has been recently attracting a lot of attentions. RL based algorithms have achieved tremendous
results in various fields such as video games[Dota](https://blog.openai.com/openai-five/), robotics[robotics](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html),
and model architecture search[[architecuter]]((https://arxiv.org/pdf/1611.01578.pdf)). Although financial
trading is also one of the potential applications, this area has been relatively less researched due
to some difficulties to deal with the dynamical market environment. In this blog post, I will overview 
some researches and try algorithms suggested at a paper for portfolio management [port](https://arxiv.org/pdf/1706.10059.pdf).


# Why we should use Reinforcement Learning
In a blog post of `WildML`[[wildml]](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)
, the author overviews the landscape of applications of RL to algorithm trading and remarks the reasons why RL is promising.

The first reason is to allow you to build end-to-end system easily. Supervised learning algorithms may predict
the price movement, but they do not tell you how to allocate your assets. That is, you need to implement another
algorithm or strategy to determine the betting size. On the other hand, RL can directly optimizes a policy that determines
optimal asset allocation.

The other reason is to make it easy to take complex dynamics of a environment into consideration.
For example, you do not know what price you can actually execute your orders at. There is what is called
slippage. Even if you want to sell or buy stocks at the price you are seeing in the market right now,
their prices are changing before all of the executions have been completed. Besides that, there is usually
transaction fee, which may eat up your account if you have too frequent transactions with small profits. 
It hard to label data for supervised learning algorithms to take all of fees and slippage into considerations.
RL does not require explicit labels for that.

Due to above reasons, RL might be promising algorithms for finance. Collecting experiences for learning,
however, could be more expensive than successful applications like video games. If you screw at a video game,
you do not lose anything. You just accumulate frustrations and waste your time. 
This is not the case for trading. You do not want to blow up your bank account just for learning your model.
Therefore, most researchers take alternative approach, using either of synthetic or historical data.


# Research so far ...
The synthetic data can be generated through mathematical models. The research team from J.P.Morgan
attempt to find hedging strategies using a neural network instead of solutions deduced from mathematical
equations. They test proposed algorithms over data generated from Heston model[[heston]](https://en.wikipedia.org/wiki/Heston_model).
Their frame work is dealing with data sequentially through RL to minimize a convex risk measure. I. Halperin
theorizes and suggests a way to implement Q-Learning and its numerical experiments for option pricing and hedge strategies [qlbs](https://arxiv.org/pdf/1712.04609.pdf).
Although He makes this point for a single risky stock and minimization of variance of a hedge portfolio, 
the approach can be extended to multiple stocks and other utility functions. The thing both approaches
have in common is that the approaches do not assume the existence of risk-neutral measure, which is often
assumed in classical mathematical finance. Besides that, they are model free approaches. You are able to
combat the complex market dynamics.

Historical data is also used to verify model performance. David W. Lu suggests an algorithm using Q-learning
to make signals for positions(long or short) [agentinspired]((https://arxiv.org/pdf/1707.07338.pdf)). He uses trailing history
of price  difference, previous position, and bias parameter for input of LSTM or RNN. He directly optimizes
directly optimize Sharp Ratio or Downside Deviation.
Z. Jiang, et al, use policy gradient to generate the percentage of allocation for each asset [[portofliomanagement]](https://arxiv.org/pdf/1706.10059.pdf).
For example,
if you have 10 currencies and one cash, output dimension will be 11 dimension. Each of element represents
percentage of allocation. Their approach is more like online learning. For each frame, the algorithm receive OHLC data and resample
data to update parameters to maximize accumulated returns over sampled data points. Optimizing accumulated
returns may more make sense in finance if you use historical data to train your model. What you have to
maximize is more obvious for each frame than video game. That would be returns and risk in finance.
Approach directly optimiznig certain cost fucntion instead of value function may more make sense in finance,
because we usually have idea about what we have to optimize, cumulative returns, Sharpe Ratio, etc.  

Both synthetic and historical data have pros and cons. Synthetic data is easy to collect, and you can test your algorithms
over lager number of sample paths to make sure statistical significance. The generated paths, however,
do not always reflect the actual property of the market dynamics. Indeed, the actual price data is not stationary.
The market is always changing the regime and price distribution. Mathematical modeling is powerful,
but models are not perfect.

Historical data allows you to simulate your model over events that actually have happened in the past.
But, the number of data points may not be large enough to train and validate models. Especially, 
most deep reinforcement learning algorithms require millions of steps to achieve good performances.

As an intermediate approach, we can consider resampling paths from historical data. H. White suggests a bootsrapping
method for sequential data to test statistical significance of trading algorithms [reality](https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf).
In chapter 12 of `Advances in Financial Machine Learning`[[advancedml]](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089), 
the author suggests split historical data into subsets and generate paths from heir combinatorics. This
approach can be seen as sampling different stories from actual sequence of events. This allows you
to see the robustness of generated models over multiple stories. For example,
if you take out data around 2008 from training dataset, you train a model that does not know the financial crisis in 2008
and see the performance when the model faces it. 

 

We have considered several methods to train and validate models so far. The thing that none of them
could cover is slippage and effect toward the market from your executions.
As mentioned previously you are not always able to execute your order immediately and the price you actually 
execute your orders at would change. Especially, if you deal with illiquid or volatile assets, this gap becomes huge.
Although we have not mentioned so far, the financial market always react to your orders.
As long as you are dealing with small amount of money from your personal bank account, the effect would be trivial.
This, however, is not the case for large financial institutes like  hedge funds and investment banks.
To find truely optimal strategies, we need to take care of these factors as well.  

To model or learn pseudo environments reflecting these factors could be one of the research direction.
I have not come across significant results for this topic so far. If you know something, let me know ;)

## Implementation
In this blog post, I will try out the algorithms suggested by [[portofliomanagement]](https://arxiv.org/pdf/1706.10059.pdf). 
I will implement PyTorch and please check out github repository.

# References
- [1] [Introduction to Learning to Trade with Reinforcement Learning](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)
- [2] [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059.pdf)
- [3] [Deep Reinforcement Learning in High Frequency Trading](https://arxiv.org/pdf/1809.01506.pdf)
- [4] [Deep Hedging](https://arxiv.org/pdf/1802.03042.pdf)
- [5] [Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks](https://arxiv.org/pdf/1707.07338.pdf)
- [6] [QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds](https://arxiv.org/pdf/1712.04609.pdf)
- [7] [Heston Model](https://en.wikipedia.org/wiki/Heston_model)
- [8] [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01578.pdf)
= [9] [Scalable Deep Reinforcement Learning for Robotic Manipulation](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)
- [10] [OpenAI Five](https://blog.openai.com/openai-five/)
- [11] [Algorithm Trading using Q-Learning and Recurrent Reinforcement Learning](http://cs229.stanford.edu/proj2009/LvDuZhai.pdf)
- [12] [A Reality Check for Data Snooping](https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf)
- [13]


# Reinforcement Learning
## Deep Reinforcement Learning in High Frequency Trading
* https://arxiv.org/pdf/1809.01506.pdf
* Ensemble three one-vs-one MLP, each of which predict binary labels
* Ensemble weight keeps updating through Reinforcement Learning
* 500 trailing tick history and label for 100 forward
* MLP has (10, 10) hidden layers
* Accuracy will be around 70 %
* Only when confidence level is over the threshold, execute trading
* The perticipant percentage is around 10%

## A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
* https://arxiv.org/pdf/1706.10059.pdf
* Use Close, High, and Low as input
* Use replay memory and portfolio vector memory for training
* Use 50 trailing history

## Deep Hedging
* https://arxiv.org/pdf/1802.03042.pdf
* Find the hedging strategy through Neural Network
* Optimize convex measure
* Show the experiments over numerically sampled Heston model
* Use (2d, d+15, d+ 15, d) MLP Model, where d is the number of assets

## Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks
* https://arxiv.org/pdf/1707.07338.pdf
* Direct reinforcement(Not required value function)
* Recurrent Reinforcement Learning using LSTM or RNN
* Optimize Sharp Ratio or Downside Deviation
* Input:
    * Trailing history of price difference
    * Previous position
    * Bias parameter
* Previous position is fed into the input of the output layer
* Validation is done on a single period, which is not trustable


## QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds
* https://arxiv.org/pdf/1712.04609.pdf
* Derive pricing through Q-Learning format
