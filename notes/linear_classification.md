---
layout: page
title: Classification
img: classification.svg
---

[Download notes as jupyter notebook](linear_classification.tar.gz)

## Introduction

Whereas our previous introduction to machine learning focused on the case of predicting real-valued outputs (called the regression setting), these notes consider the case of predicting discrete-valued outputs.  This is called the classification setting.  And while it may seem as though this would require very different techniques to handle (after all, the regression setting we just considered made very "continuous" predictions rather than discrete ones), one nice thing about our generic framework for machine learning is that in reality, there is _very_ little difference between how we handle regression and classification problems.  Like before, there are hypothesis functions, loss functions, and our goal is still to solve the canonical problem of minimizing the sum of losses on a data set (the only real difference is in the choice of loss function, as we will see even the hypothesis class largely remains the same).

## Example: breast cancer classification

As an example to introduce classification problem, we're going to turn to a classical example from the machine learning literature, first presented in [[Street et al, 1993]](http://ftp.cs.wisc.edu/math-prog/tech-reports/olm102.ps) of identifying whether cancer cells from a biospy are malignant or benign.  This was historically an important data set in the intersection of machine learning and health care, and similar subjects are still a topic of very active research.  Of course, 25 years ago, the techniques were much more primitive and computationally less intense than current approaches to similar problems, but the work has become a classical data set in machine learning, and also represent a compelling applications.

To create the data set, researchers took 569 images of cancerous cells, under a microscope, and manually selected the outlines of the different cells (this step is the kind of thing that would ideally be replaced by automatic computer vision architectures in current systems).  The resulting figures looked like this:
{% include image.html img="cancercells.png" caption="Image of (manually) selected outlines of cells under a microscope."%}
The researchers then considered 10 different features of each cell, of instance the area, perimeter, texture, number of concave points (i.e., indentations), variance of grayscale color, and some others (a full list is in the above paper).  For each of these 10 features they computed the mean, max, and standard deviation of the feature across all the cells in the image, resulting in a 30 dimensional vector describing each sample.

The following is a plot of two of these features (mean # of concave points, and mean area) for all examples in the data set; blue points denotes a benign tumor, whereas red points denote a malignant tumor.  (Note that the dataset is so common, it's included in the actual scikit-learn package itself).


```python
import numpy as np
import sklearn.datasets
dat = sklearn.datasets.load_breast_cancer()

def plot_cells():
    ix = np.where(dat["feature_names"] == "mean area")[0][0]
    iy = np.where(dat["feature_names"] == "mean concave points")[0][0]
    plt.scatter(dat["data"][:,ix][dat["target"]==1], dat["data"][:,iy][dat["target"]==1], marker='x', color='C0')
    plt.scatter(dat["data"][:,ix][dat["target"]==0], dat["data"][:,iy][dat["target"]==0], marker='+', color='C3')
    plt.xlim([0,2600])
    plt.ylim([0,0.21])
    plt.xlabel("Mean Area")
    plt.ylabel("Mean Concave Points")
    plt.legend(['Benign', 'Malignant'])
plot_cells()
```


{% include image.html img="output_0.svg" %}


```python
np.where(dat['target_names']=="malignant")[0][0]
```

<pre>
0
</pre>

There is obviously some structure to the data here: cells with greater average area and greater numbers of concave points are more likely to be malignant.  And whereas your mental model of simple linear regression should be something like "fitting a line to the data", your mental model of a classification task should be "separating the classes with a line".  Something like the following (just a hand-picked example of a line, we will shortly see how to actually fit these to the data, much like we did for regression).


```python
plot_cells()
plt.plot([0, 1300], [0.08,0], 'k-')
```


{% include image.html img="output_1.svg" %}

Just as our linear regression line did not pass exactly through all the data points, we can't achieve here a line that exactly separates all the data points.  But there is clearly as sense in which points below the line are "more likely" to be benign, and above the line are "more likely" to be malignant.  To make this more formal, we are going to introduce the notation of machine learning classification.

## Classification notation

As a reminder, let's consider the basic elements of a machine learning setup, available in more detail [here](http://www.datasciencecourse.org/notes/ml_intro/#machine-learning-in-general).

- **Inputs (features):** $x^{(i)} \in \mathbb{R}^n, \; i=1,\ldots, m$
- **Outputs:** $y^{(i)} \in \mathcal{Y}, \; i=1,\ldots,m$
- **Parameters:** $\theta \in \mathbb{R}^d$.
- **Hypothesis function:** $h_\theta : \mathbb{R}^n \rightarrow \hat{\mathcal{Y}}$
- **Loss function:** $\ell : \hat{\mathcal{Y}} \times \mathcal{Y} \rightarrow \mathbb{R}\_+$
- **Canonical optimization problem:**

$$
\DeclareMathOperator*{\minimize}{minimize}
\minimize_\theta \frac{1}{m}\sum_{i=1}^m \ell(h_\theta(x^{(i)}),y^{(i)})
$$


In the context of classification, these are mostly unchanged.  The one difference is that the output space, $\mathcal{Y}$, is now a discrete space instead of a continuous one: $\mathcal{Y} = \\{-1,+1\\}$ for binary classification, or $\mathcal{Y} = \{1,2,\ldots,k\}$ for multiclass classification.  For the remainder of these notes, unless otherwise we're going to assume the binary setting, but it's important to note that our convention will be that $\mathcal{Y} = \{-1,+1\}$ _not_ $\mathcal{Y} = \{0,1\}$ (this will make a difference, since we're going to use the numerical values of $y$ is our loss functions.

The other notable difference is that predicition space (the output of the hypothesis function) $\hat{\mathcal{Y}}$ is actually _still_ the same as it was in the regression setting, $\hat{\mathcal{Y}} = \mathbb{R}$ (again, this assumes binary classification, and we'll make a slight change for the multiclass setting).  This may seem somewhat counter-intuitive: we have outputs in the discrete space $\{-1,+1\}$ and predictions that are real-valued.  Intuitively, the output of our hypothesis function will correspond to some notion of "confidence" in our prediction: the _sign_ of $h_\theta(x)$ determines the most likely output (we are predicting positive if $\mathrm{sign}(h_\theta(x)) = 1$ and negative if $\mathrm{sign}(h_\theta(x)) = -1$), but the magnitude indicates how certain we are: large positive values of $h_\theta(x)$ indicate we are very confident in our positive prediction; very large negative values indicate we are very confident in our negative prediction; and values around zero indicate we are not particularly certain one way or another in our prediction.  Without these notions of confidence, it is extremely difficult to formulate a "smooth" loss that can be optimized with, gradient descent, for example.

The loss functions are the one element that _is_ substantially different between classification and regression, and we will talk about these moment, but first describe how the other elements are instantiated in our cancer classification task.

### Instantiation in cancer domain

In the setting above, predicting benign or malignant cells from the two features shown (here just considering predictions based upon these two features, not all 30 features that are in the dataset), we would set up the problem as:

- **Inputs (features):** $\displaystyle x^{(i)} \in \mathbb{R}^3 = 
\left[\begin{array}{c}\mathrm{MeanArea}^{(i)} \\
\mathrm{MeanConcavePoints}^{(i)} \\ 1 \end{array} \right ]$, where again, we add the constant term 1 to integrate the intercept function directly into the inputs.
- **Outputs:** $y^{(i)} \in \{-1,+1\}$ where $y^{(i)} = -1$ corresponds to a benign tumor and $y^{(i)} = +1$ corresponds to a malignant tumor.  Note that these labels could be switched of course, but for the sake of simplicity we typically try to choose the labels to correspond to our common notions of "positive" and "negative" examples. 
- **Parameters:** $\theta \in \mathbb{R}^3$.  These are the same as before, as we are going to use a linear hypothesis function.
- **Hypothesis function:** We use a linear hypothesis function, $h_\theta : \mathbb{R}^3 \rightarrow \mathbb{R}$, $h_\theta(x) = \theta^T x = \sum_{j=1}^3 \theta_j x_j$.  Again, for classification this will not correspond to a line that goes "through" the data as it did for linear regression, but to a "hyperplane" that is positive in the region where we are predicting positive and negative where we are predictive negative.  The decision boundary, the "line" that _separates_ the data, is given by the equation $h_\theta(x) = 0$.

## Loss functions in classification

How do we define a suitable loss function for classification $\ell : \hat{\mathcal{Y}} \times \mathcal{Y} \rightarrow \mathbb{R}\_+$?  This is not completely obvious, because our prediction space is continuous $\hat{\mathcal{Y}} = \mathbb{R}$, while our output space is discrete $\mathcal{Y} = \{-1,+1\}$.  Before defining the loss functions that are actually used in practice in classification, let's consider one "easier" alternative:

### Least squares classification
One of the more obvious approaches we could employ is the following: just use the squared loss (or some regression loss) for our classification task.  In other words, while it is true that the $y^{(i)}$ terms are discrete, we can just treat them as the real-valued numbers $-1$ and $+1$, and use a loss like $\ell(h_\theta(x),y) = (h_\theta(x) - y)^2$.  Despite the fact that I'm about to argue against this approach, it turns out this is not at all a bad idea in practice, and it absolutely makes sense to run this approach (referred to sometimes as "least squares classification") as a first take on classification tasks, especially since we can use the analytic least squares solution.

However, why might this sometimes be a bad idea?  Let's consider the following data set:
{% include image.html img="classification_1.svg" caption="An example one-dimensional classification task."%}

If we fit a line to this data using least squares, we would get something like the following:
{% include image.html img="classification_2.svg" caption="A least-squarse fit to our one-dimensional problem."%}

Remember that in the context of classification, we would be predicting a positive class wherever the line is positive, and a negative class where the line is negative.  But this means that we actually predict incorrectly on the left-most positive point: our hypothesis here is negative while the label is positive.  On the other hand, it's of course completely possible to classify the data perfectly with a linear classifier, as in the figure below:
{% include image.html img="classification_3.svg" caption="A perfect (linear) classifier on the one-dimensional problem."%}

The issue here, of course, is that the least-squares loss function applied to classification aims at predicting _exactly_ $+1$ or $-1$ on each data point.  There is no sense in which it is "ok" to have a number much larger than one for a positive example, whereas of course this is completely acceptable in the classification setting.  And by the nature of the least squares loss (the same is also true even for robust losses, though to a lesser degree), these "outliers" will add greatly to the loss function, and can substantially change the resulting classification of many points.

Motivated by this concern, let's discuss alternative loss functions for classification.

### Zero-one loss

Perhaps the most obvious loss, which is frankly what people just assume to be the "correct" loss for classification problems, is just the number of mistakes that the classifier makes.  That is, if the sign of $h_\theta(x^{(i)})$ is negative while $y^{(i)}$ is positive (or vice versa), this counts as a mistake, whereas the point is correctly classified otherwise.  This is known as the zero-one loss $\ell_{0/1} : \mathbb{R} \times \{-1,+1\} \rightarrow \mathbb{R}\_+$, and we can write it as follows

$$
\begin{split}
\ell_{0/1}(h_\theta(x), y) & = \left \{ \begin{array}{ll} 0 & \mbox{ if } \mathrm{sign}(h_\theta(x)) = y \\
1 & \mbox{ otherwise } \end{array} \right . \\
& = \mathrm{1}\{h_\theta(x)\cdot y \leq 0\}.
\end{split}
$$

This last way of writing the quantity is important (here $\mathrm{1}\{\cdot\}$ denotes the zero-one indicator function,  a function that is equal to one if its inner argument is true, and zero otherwise).  Specifically, what is important to consider here is the quantity $h_\theta(x)\cdot y$, the hypothesis times the output.  First, you should convince yourself that the two definitions of the zero-one loss are in fact equivalent: $h_\theta(x)$ and $y$ have the same sign (either positive or negative), then $h_\theta(x)\cdot y$ will be positive, whereas if $h_\theta(x)$ and $y$ have difference signs, then $h_\theta(x)\cdot y$ will be negative, and so incur a loss of one.  But the important notion here is that, with our definition that $h_\theta(x)$ be a measure of how confident we are in our prediction, $h_\theta(x)\cdot y$ will be positive if we are "confidently correct" and very negative if we are "confidently incorrect" (we have a prediction with large magnitude, but which is incorrect).

Let's consider a plot of the zero-one loss below.


```python
hy = np.linspace(-3,3,1000)
plt.plot(hy,(hy<=0))
plt.xlim([-3,3])
plt.ylim([-0.05, 2])
plt.ylabel("Loss")
plt.xlabel("$h_θ(x) \cdot y$")
```


{% include image.html img="output_2.svg" %}

Note that the $x$ access here is precisely the quantity we discussed $h_\theta(x)\cdot y$, whereas the $y$ axis shows the loss we suffer.  This is the common way of visualization (binary) classification losses, as it shows how much loss is suffered as the prediction gets more confidently correct or incorrect (of course, for zero-one loss there is no notion of confidence).

While zero-one loss is very natural to consider for classification problems, it has two main challenges: first, it is difficult to optimize.  We won't show it here, but it turns out that in general it is NP-hard to find even a linear classifier that minimizes the zero-one loss for a general collection of points.  The second issue is that the zero-one loss is non-smooth, and has derivative equal to zero everywhere except for the zero point, where the derivative is undefined.  This makes it very difficult to even _approximately_ minimize the total zero-one loss (with techniques like gradient descent).  Thus, we typically want to consider alternative loss functions for classification tasks.

### Alternative losses for classification

Given the difficulties of minimizing sum of zero-one losses, one typically uses an alternative loss in classification.  There are many such losses, and different choices here lead to different machine learning algorithms (often with very little indication that all that differs between the algorithm is in fact the choice of loss function).  Let's consider a few of of the more common ones:

- **Logistic loss:** $\ell_{\mathrm{logistic}}(h_\theta(x), y) = \log(1+\exp(-h_\theta(x) \cdot y))$  <br/> For large positive values of $h_\theta(x) \cdot y$, $\exp(-h_\theta(x) \cdot y)$ will be very close to zero, so the loss will approach zero; for large negative values $\log(1+\exp(-h_\theta(x) \cdot y)) \approx -h_\theta(x) \cdot y$ so the loss increases approximately linearly.
- **Hinge loss:** $\ell_{\mathrm{hinge}}(h_\theta(x), y) = \max\{1 - h_\theta(x) \cdot y, 0\}$ <br/>  As long as $h_\theta(x) \cdot y \geq 1$, this loss will be zero, whereas it will increase linearly for negative $h_\theta(x) \cdot y \geq 1$.
- **Exponential loss:** $\ell_{\mathrm{exp}}(h_\theta(x), y) = \exp(-h_\theta(x) \cdot y)$ <br/> For a similar reason as with logistic loss, this loss will go to zero for large $h_\theta(x) \cdot y$; but for negative $h_\theta(x) \cdot y$ the loss increases very quickly.

Let's plot some of these loss functions relative to zero-one loss.


```python
hy = np.linspace(-3,3,1000)
plt.plot(hy,(hy<=0))
plt.plot(hy, np.log(1+np.exp(-hy)))
plt.plot(hy, np.maximum(1 - hy, 0))
plt.plot(hy, np.exp(-hy))
plt.xlim([-3,3])
plt.ylim([-0.05, 5])
plt.ylabel("Loss")
plt.xlabel("$h_θ(x) \cdot y$")
plt.legend(['Zero-one', 'Logistic', 'Hinge', 'Exponential'])
```


{% include image.html img="output_3.svg" %}

One thing that notably differs between these losses is the scale to which they are affected by outliers (in the classification context, meaning points that have high magnitude hypothesis, but which are classified incorrectly, i.e., the "confidently incorrect" points).  Zero-one loss is the least sensitive to such outliers, as it suffers the same loss regardless of how confident the incorrect prediction is.  Exponential loss is on the other extreme, with a loss function that increases extremely quickly for predictions that confidently incorrect.  And logistic and hinge loss lie somewhere in between, with a loss that increases _linearly_ with the confidence of incorrect predictions.  In fact, logistic loss and hinge loss are extremely similar in this regard, with the primary difference being that the logistic loss is continuously differentiable and always strictly positive, whereas the hinge loss has a non-differentiable point at one, and is exactly zero beyond this point.

### Solving classification tasks

With this framework and set of loss functions laid out, the task of learning a classifier is exactly the same as the regression methods we considered previously.  That is, given a data set we seek to minimize the average loss

$$
\minimize_\theta \frac{1}{m}\sum_{i=1}^m \ell(h_\theta(x^{(i)}),y^{(i)}).
$$


Unlike regression, there is virtually no setting (except, of course, just using least-squares loss) where there is an analytical solution to this optimization problem.  Nonetheless, we can still solve it by resorting back to gradient descent (there are alternative optimization approaches for most of these problems, but we'll only cover gradient descent).  To see how this works, we'll next consider some collection of specific machine learning algorithms that result in specific choices of loss function.

## Example classification algorithms

Let's consider a few specific cases of machine learning algorithms, which really just result from different choices of loss functions.  In all cases, we'll simply present the loss and derive the gradient, then illustrating how to perform the gradient descent algorithm.

### Support vector machines

Support vector machines (SVMs) result from choosing hinge loss

$$
\ell_{\mathrm{hinge}}(h_\theta(x),y) = \max \{1-h_\theta(x)\cdot y, 0\}
$$

as the loss function to minimize.  If you have seen support vector machines before, you may have seen them explained in a geometric context, illustrating that they maximize the "margin" of separation between examples of different classes.  This is true, and support vector machines absolutely can be viewed this way.  But it is also instructive to simply understand them as minimizing the particular loss function above.

Support vector machines also refer either to a linear hypothesis function, or a particular form of nonlinear hypothesis known as a _kernel_ hypothesis; if you ever hear the terms "linear SVM" or "kernel SVM", this is just the designation of which type of hypothesis function they are using.  That is, linear SVMs use the hypothesis function

$$
h_\theta(x) = \sum_{j=1}^n \theta_j x_j = \theta^T x,
$$

exactly as we have seen before in the linear regression case.  While we won't discuss it in any detail here (and we will only briefly discuss it when we talk about nonlinear hypotheses next lecture), very briefly, a kernel SVM uses the hypothesis function

$$
h_\theta(x) = \sum_{i=1}^m \theta_i K(x,x^{(i)})
$$

where $\theta \in \mathbb{R}^m$ are the parameters, and $K : \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$ is known as a _kernel function_.  The interesting elements to note here are the fact that $\theta$ now is the same size as the _number of data points_ (that is, it is $m$ dimension, this is not a typo), and the prediction on some new point $x$ requires evaluating the kernel function $K(x, x^{(i)})$ between the input to our hypothesis $x$ and all our data points.  This means that the complexity kernel approaches will scale with the number of datapoints instead of the dimension of the input.  Again, we aren't describing this here in any detail, but you should be aware of some of these big ideas.  For now, we will only consider the case of the linear SVM, so we're going to assume a linear hypothesis for the remainder of this section.

In addition to the hinge loss term above, A little bit more correctly, support vector machines actually include another term besides just the average loss in their formulation: a _regularization_ term that penalizes the (squared) magnitude of the sum of weights $\sum_{j=1}^n\theta^2_j$.  We're going to cover regularization much more next lecture, so we won't address this point any more for now, except to say that it has the effect of balancing out the "complexity" of the classifier (which increases with larger weights) with the goal of minimizing the loss.  With this additional term, the full optimization problem considered by a (linear) support vector machine is given by

$$
\minimize_{\theta} \frac{1}{m} \sum_{i=1}^m \max \{1-\theta^T x^{(i)} \cdot y^{(i)}, 0\} + \lambda \sum_{j=1}^n \theta_j^2
$$

where $\lambda \in \mathbb{R}\_+$ is what is called a _regularization parameter_ that trades off between the goal of minimizing the loss and keeping the weights small.

Let's derive the gradient descent algorithm for linear SVMs.  Recall that the hinge loss looks like the following, as a function of $h_\theta(x) \cdot y$:


```python
hy = np.linspace(-3,3,1000)
plt.plot(hy, np.maximum(1 - hy, 0))
plt.xlim([-3,3])
plt.ylim([-0.05, 5])
plt.ylabel("Loss")
plt.xlabel("$h_θ(x) \cdot y$")
```


{% include image.html img="output_4.svg" %}

Thus, the partial derivative of the loss is given by

$$
\frac{\partial}{\partial \theta_j} \max \{1-\theta^T x \cdot y, 0\} = -x_j y 1\{\theta^T x \cdot y \leq 1\},
$$

the gradient of the regularization term is given by

$$
\frac{\partial}{\partial \theta_j}\lambda \sum_{k=1}^n \theta_k^2 = 2\lambda \theta_j
$$

and so the gradient of the entire objective is given by

$$
\nabla_\theta E(\theta) = 
\nabla_\theta \left (\frac{1}{m} \sum_{i=1}^m \max \{1-\theta^T x^{(i)} \cdot y^{(i)}, 0\} +\lambda \sum_{k=1}^n \theta_k^2\right ) = \frac{1}{m} \sum_{i=1}^m -x^{(i)} y^{(i)} 1\{\theta^T x^{(i)} \cdot y^{(i)} \leq 1\} + 2\lambda \theta.
$$


We can write this efficiently in matrix format (using the same matrix definitions as we used in the previous lecture), as 

$$
\nabla_\theta E(\theta) = -\frac{1}{m}X^T Y 1\{Y X\theta \leq 1\} + 2 \lambda \theta
$$

and where $Y = \mathrm{diag}(y)$ is a diagonal matrix made from the elements of $y$ (not that you wouldn't actually from this matrix, this just scales the rows of $X$ by $y$.  Here is what this looks like in code.


```python
def svm_gd(X, y, alpha, iters, lam):
    theta = np.zeros(X.shape[1])
    YX = X * y[:,None]
    loss, err = np.zeros(iters), np.zeros(iters)
    for t in range(iters):
        hy = YX @ theta
        loss[t] = np.maximum(1-hy,0).mean()
        err[t] = (hy <= 0).mean()
        theta -= alpha * (-YX.T @ (YX @ theta <= 1)/X.shape[0] + 2*lam*theta)
    return theta, loss, err
```

Again, the actual nuts and bolts of a complex seeming ML algorithm boils down to something fairly simple.  Let's apply this to our example above, where we now go through the normal routine of normalizing the data, etc (note that for a  classification tasks, we naturally don't normalize the output).


```python
def normalize_data(X, normalize_cols):
    """ Normalize y and specified columns of X in place. """
    min_X = X[:,normalize_cols].min(axis=0)
    max_X = X[:,normalize_cols].max(axis=0)
    X[:,normalize_cols] = (X[:,normalize_cols] - min_X) / (max_X - min_X)
    return min_X, max_X

def unnormalize_theta(theta, normalize_cols, ranges):
    theta[normalize_cols] /= (ranges[1] - ranges[0])
    theta[-1] -= theta[normalize_cols] @ ranges[0]
```


```python
# build the data and run the SVM
X = np.asarray([dat["data"][:,ix], dat["data"][:,iy], np.ones(dat["data"].shape[0])]).T
y = (1.*(dat["target"]==np.where(dat["target_names"] == "malignant")[0][0]) +
     -1.*(dat["target"]==np.where(dat["target_names"] == "benign")[0][0]))
ranges = normalize_data(X, [True, True, False])
theta, loss, err = svm_gd(X,y, 1.0, 1000, 1e-4)
unnormalize_theta(theta, [True, True, False], ranges)
print(theta)
```

<pre>
[  2.61807438e-03   3.94345987e+01  -3.89894017e+00]
</pre>

Let's first visualize what this classification looks like.  Note that here we are using the `contour` functions to draw the positive and negative regions; for a simple linear classifier there are easier ways, but the approach here will also work for nonlinear classifiers with appropriate modifications.


```python
def plot_classification(theta):
    x_range = [0,2600]
    y_range = [0, 0.21]

    # plot classifcation regions
    grid=200
    xx,yy = np.meshgrid(np.linspace(x_range[0], x_range[1], grid),
                        np.linspace(y_range[0], y_range[1], grid))
    X_ = np.array([np.ravel(xx), np.ravel(yy), np.ones(xx.size)]).T
    zz = (X_ @ theta).reshape(grid,grid)
    cs = plt.contourf(xx,yy,zz, zz,levels=[-100.0,0,100.0],alpha=0.2,colors=["b","r"])
    plt.contour(cs, colors='k')

    ix = np.where(dat["feature_names"] == "mean area")[0][0]
    iy = np.where(dat["feature_names"] == "mean concave points")[0][0]
    s1 = plt.scatter(dat["data"][:,ix][dat["target"]==1], dat["data"][:,iy][dat["target"]==1], marker='x', color='C0')
    s2 = plt.scatter(dat["data"][:,ix][dat["target"]==0], dat["data"][:,iy][dat["target"]==0], marker='+', color='C3')
    plt.xlim([0,2600])
    plt.ylim([0,0.21])
    plt.xlabel("Mean Area")
    plt.ylabel("Mean Concave Points")
    plt.legend([s1,s2], ['Benign', 'Malignant'])
```


```python
plot_classification(theta)
```


{% include image.html img="output_5.svg" %}

(to be continued)

