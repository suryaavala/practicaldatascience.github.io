---
layout: page
title: Basics of probability
img: probability.svg
---

[Download notes as jupyter notebook](probability.tar.gz)

** Notes will be updated/finalized by 3/26 **

## Introduction

Up until this point in the course, we have largely ignored formal probabilistic statements or formal definitions of probability.  This may seem somewhat surprising: the topic of probability is naturally tightly intertwined with data science and machine learning, and it may seem odd to have gotten so deep into these topics (it is also not quite true, since we did use probability in an informal way when discussion n-gram models for free text).  But we previously largedly considered machine learning from an optimization context, in terms of minimizing loss function, and even our presentation of generalization and overfitting only mentioned probability informally.

From this point on, this will no longer be the case.  While it is useful to understand machine learning from the optimization standpoint, machine learning as a field is also deep tied to probability and statistics, and the goal of these notes is to provide a basic introduction to some of the principles behind probability.  As a concrete example of why understanding and modeling probabilities can be so important in data science (this is of course just one example), let's return to considering previous example of high temperature vs. peak demand.


```python
df = pd.read_csv("temp_demand.csv").set_index("Date")
df_summer = df[list(map(lambda x : str(x)[4:6] in ["06", "07", "08"], df.index))].copy()
plt.scatter(df_summer["Temp"], df_summer["Load"], marker='x')
plt.xlabel("High Temperature")
plt.ylabel("Peak Demand")
```


{% include image.html img="output_0.svg" %}

Although there _is_ a clear linear pattern to this data, there is also the sense in which we _don't_ expect to be able to predict the peak demand exactly.  After all, peak electrical demand is something that _I_ even have some control over (I could choose to turn a lightswitch on or off tomorrow, which would affect the peak demand, albeit in a very limited manner).  So while we can reduce the uncertainty somewhat (remember, for example, that including a "weekday" versus "weekend" feature did exactly this), we don't expect to eliminate the uncertainty completely, at least not at the level that we are ever going to be able to observe the data.  This is a critical point: _many machine learning tasks cannot be "solved" in the sense of getting perfect predictions, but the correct solution is instead just to accurately quantify the distribution over possible outputs_.

With this in mind, we're going to begin our discussion on basic probability, with the next section additionally covering some basic statistics.  This distinction is actually quite important, and for the purposes of this course you can think of it as the following: probability covers the basic rules of random variables manipulating expressions related to them, while statistics involves the incorporation of data (i.e., fitting probability distributions to observations).  These notes will be covering the former category, understanding the rules behind random variables and how we make statements about them.

## Random Variables

The basic building block of probability is the random variable.  If you take a more formal course in probability theory, you will see a much more rigorous definition of a random variable (i.e., random variables as functions), and even some introductory courses describe them from a more formal standpoint (talking about outcome spaces, the axioms of probability, etc).  In this course (because this is just a single lecture, after all), we're going to eschew all this formality, and present random variables at the level that I believe to be intuitive and useful for data science.  In particular, the main thing you'll do with probabilities in data science tasks is use some basic properties of probability to manipulate probabilistic expressions, and describe random variables using some basic well-known distributions.  We'll talk about both these elements here.

For our purposes, you can think of a random variable as just a variable "whose value is not yet known".  Rather, each value that the variable may take on has some probability of occuring (yes, this is a circular definition, but we're presuming you have an intuitive notion of probability at this level).  For example, "Weather" could be a random variable representing the weather that will occur tomorrow, which can take values in {"sunny", "rainy", "cloudy", "snowy"), each with some associated probability

$$
\begin{split}
p(\mbox{Weather} = \mbox{sunny}) & = 0.3\\
p(\mbox{Weather} = \mbox{rainy}) & = 0.2\\
\vdots
\end{split}
$$


We'll need a slightly different notation for continuous random variable, which we'll discuss later.

### Notation for random variables

One of the hardest parts about understanding what is meant by certain notation.  And we fully admit that the notation for probability _is_ often quite confusing, and seems somewhat odd at first.  But underlying this are some fairly simple rules, which we will stick to at least for this set of notes.  The potentially annoying aspect is that this notation can and does change when we shift back to talk about machine learning, for example (and we will also change the notation back to be what is common in those areas), but hopefully at that point you'll have a sufficient understanding of the topic to understand what is meant in different situations.

In general, we will use captial letters, $X$ to represent random variables.  Or if the random variable is a quantity like weather, we will use the capitalized $\mbox{Weather}$ to denote the random variable.  For a random variable $X$ taking on values, say in $\\{1,2,3\\}$, the notation

$$
p(X)
$$

represent the _entire probability distribution_ of the random variable $X$.  In other words, you should think of $p(X)$ really as representing a function mapping from the different values that $X$ can take on to their associated probabilities.  In Python, you could implement this as a dictionary for instance.


```python
pX = {1: 0.1, 2: 0.5, 3:0.4}
pWeather = {"sunny": 0.3, "rainy":0.2, "cloudy":0.4, "snowy":0.1}
```

This notation $p(X)$ is somewhat unfortunate, because $p(X)$ does not really represented a function _of_ $X$, but an entire function mapping values that $X$ takes on to positive numbers (which need to sum to one).  Yet we also won't use notation like $p(X)(1)$ either, but instead using something like $p(X=1)$, so this seems a bit ambiguous.  A better notation would be something like $p_X$, but this is less common, so we'll stick to the usual $p(X)$, as long as you understand that this really refers to the entire distribution.

If we _do_ want to refer to a specific value that $X$ takes on, we'll use the lowercase $x$, i.e., in the above case to represent one of the values that $X$ takes on, $x \in \\{1,2,3\\}$.  When then use the notation

$$
p(X=x) \;\; \mbox{or just} \;\; p(x)
$$

to represent the probability (i.e., the numerical value) of that particular probability.  Again in code, this would look something like the following.


```python
pX[1]
pWeather["sunny"]
```

<pre>
0.3
</pre>

### Joint distributions and factors

Given two random variables $X_1$ (taking values in $\\{1,2,3\\}$) and $X_2$ (taking values in $\\{1,2\\}$), their _joint distribution_, written $p(X_1,X_2)$ is a probability distribution mapping all possible values that both variables can take on to the respective probability.  For example, again in Python:


```python
pX1X2 = {(1,1): 0.3, (1,2):0.1, (2,1):0.05, (2,2): 0.25, (3,1):0.15, (3,2):0.15}
print(pX1X2)
```

<pre>
{(1, 1): 0.3, (1, 2): 0.1, (2, 1): 0.05, (2, 2): 0.25, (3, 1): 0.15, (3, 2): 0.15}
</pre>

Then as above, $p(x_1, x_2)$ would represent a _number_ corresponding to a single entry in this distribution (the probability that $X_1 = x_1$ and $X_2 = x_2$).


```python
pX1X2[(1,2)]
```

<pre>
0.1
</pre>

Where this gets a bit tricky in when we _combine_ both types of expressions.  For example $p(X_1,x_2)$ would correspond to the entries in the above probability take for _all_ values of $X_2$, but only for a particular value of $X_2$.  In other words, it would be a function mapping from values in $X_1$ to real numbers.  In Python, this would essentially reduce to creating a new dictionary over all values of $X_1$, for a particular assignment to $x_2$.  Importantly, however, $p(X_1, x_2)$ is _not_ a true probability distribution over $X_1$, because the values do not sum to one.  Instead, these terms are generally referred to as _factors_, which you can think of like probability distributions (mappings from variables assignments to values), except that their entries need not sum to one.


```python
fX1 = {k[0]:v for k,v in pX1X2.items() if k[1] == 2}
fX1
```

<pre>
{1: 0.1, 2: 0.25, 3: 0.15}
</pre>

As a more concrete example, let's consider two random variables: $\mbox{Weather}$ (the weather tomorrow, for now taking values in $\\{\mbox{sunny}, \mbox{rainy}, \mbox{cloudy}\\}$ ) and $\mbox{Cavity}$ (whether or not I have a cavity, taking values in $\\{\mbox{yes}, \mbox{no}$); there are some old adages about people being able to predict the weather from pain in their teeth, so maybe there is some relation between these variables.  Let's first look at the complete joint distribution:

$$
p(\mbox{Weather}, \mbox{Cavity}) = \left \{ 
\begin{array}{llr}
\mbox{sunny} & \mbox{yes} & 0.07 \\
\mbox{sunny} & \mbox{no} & 0.63 \\
\mbox{rainy} & \mbox{yes} & 0.02 \\
\mbox{rainy} & \mbox{no} & 0.18 \\
\mbox{cloudy} & \mbox{yes} & 0.01 \\
\mbox{cloudy} & \mbox{no} & 0.09
\end{array} \right .
$$


If we want to refer to a single entry, we just have

$$
p(\mbox{Weather} = \mbox{sunny}, \mbox{Cavity} = \mbox{yes}) = 0.07.
$$


Or if we want to look at a partial factor

$$
p(\mbox{Weather}, \mbox{Cavity} = \mbox{yes}) = \left \{ 
\begin{array}{lr}
\mbox{sunny} & 0.07 \\
\mbox{rainy} & 0.02 \\
\mbox{cloudy} & 0.01 \\
\end{array} \right .
$$


Again as Python code:


```python
pWeatherCavity = {("sunny", "yes"):0.07,
                  ("sunny", "no"):0.63,
                  ("rainy", "yes"):0.02,
                  ("rainy", "no"):0.18,
                  ("cloudy", "yes"):0.01,
                  ("cloudy", "no"):0.09}
pprint.pprint(pWeatherCavity)
print(pWeatherCavity[("sunny", "yes")])
fWeather = {k[0]:v for k,v in pWeatherCavity.items() if k[1] == "yes"}
pprint.pprint(fWeather)
```

<pre>
{('cloudy', 'no'): 0.09,
 ('cloudy', 'yes'): 0.01,
 ('rainy', 'no'): 0.18,
 ('rainy', 'yes'): 0.02,
 ('sunny', 'no'): 0.63,
 ('sunny', 'yes'): 0.07}
0.07
{'cloudy': 0.01, 'rainy': 0.02, 'sunny': 0.07}
</pre>

### Probability/factor operations

The last topic to discuss if we we perform operations on probabilities/factors.  That is, for instance if $X_1$, $X_2$, and $X_3$ are all random variables (for simplicity say they are all binary, so taking values in $\\{0,1\\}$), then what does the following operation signify

$$
p(X_1,X_2)p(X_2,X_3)?
$$


The above product would actually be a _new_ factor over all the variables involved in term (that is, a factor over $X_1,X_2,X_3$) whose values are found by just substituting the values into each expression.  Calling this factor $f$ (we typically won't name factors, because we don't need to write them explicitly, and we just do it here to signify the notation)

$$
f(X_1 = x_1, X_2 = x_2, X_3 = x_3) = p(X_1 = x_1, X_2 = x_2)p(X_2 = x_2,X_3 = x_3)
$$


This seems confusing initially, but it is actually probably what you had in mind intuitively anyway.  Put another way, $p(X_1,X_2)p(X_2,X_3)$ is a factor where for any set of values $x_1,x_2,x_3$, to compute the value $p(x_1,x_2)p(x_2,x_3)$; and if you want to compute the full underlying factor (the value for all possible assignments of the variables), you just compute this term for all possible assignments to $x_1, x_2, x_3$.

The following code shows how to compute a factor product like this in Python.  We use the `itertools.product` call to iterate over all combinations of terms in the factors, multiply together the probability values, but then filter out only those elements where the second key of the first product ($x_2$ in the first probability) equals the first key of the second product ($x_2$ in the second probability).


```python
pX1X2 = {(0,0):0.1, (0,1):0.2, (1,0):0.3, (1,1):0.4}
pX2X3 = {(0,0):0.2, (0,1):0.3, (1,0):0.3, (1,1):0.2}

f = {(k1[0], k1[1], k2[1]):v1*v2
     for (k1,v1),(k2,v2) in itertools.product(pX1X2.items(),pX2X3.items())
     if k1[1] == k2[0]}
pprint.pprint(f)
```

<pre>
{(0, 0, 0): 0.020000000000000004,
 (0, 0, 1): 0.03,
 (0, 1, 0): 0.06,
 (0, 1, 1): 0.04000000000000001,
 (1, 0, 0): 0.06,
 (1, 0, 1): 0.09,
 (1, 1, 0): 0.12,
 (1, 1, 1): 0.08000000000000002}
</pre>

### (Aside) A cleaner Python implementation

The above notation (directly representing probability distributions as dictionaries), gets somewhat cumbersome.  When we wanted to multiply two distributions, we needed to manually remember which variable corresponded to which key in each factor.  Fortunately, we can pretty easily build a class that automates all of this, explicitly storing both the variables and their values as a complete Python dictionary.  Note that without some very heavy optimization, I would not recommend using this class directly, because it always represents factors explicitly in terms of all of their full joint distribution, but as an illustrative tool, it very useful.


```python
import operator

class Factor:
    def __init__(self, *args):
        self.factors = {(k if isinstance(k, tuple) else (k,)):v for k,v in args[-1].items()}
        self.variables = {var:tuple(set(k[i] for k in self.factors.keys())) for i,var in enumerate(args[:-1])}
    
    def __str__(self):
        string = "Factor(" + ", ".join(self.variables.keys()) + ") = {\n"
        for k,v in self.factors.items():
            string += "  " 
            string += ", ".join(["{} = {}".format(var, k[i]) for i,var in enumerate(self.variables)]) 
            string += ": {}\n".format(v)
        string += "}"
        return string

    def __repr__(self):
        return self.__str__()
    
    def __call__(self, **kwargs):
        return self.factors[tuple(kwargs[var] for var in self.variables)]
    
    def op(self, op, f):
        new_vars = dict(self.variables, **f.variables)
        vals = (dict(zip(new_vars,v)) for v in list(itertools.product(*new_vars.values())))
        return Factor(*new_vars.keys(), {tuple(k.values()):op(self(**k),f(**k)) for k in vals})
    
    def __mul__(self, other): return self.op(operator.mul, other)
    def __truediv__(self, other): return self.op(operator.truediv, other)
    def __add__(self, other): return self.op(operator.add, other)
    def __sub__(self, other): return self.op(operator.sub, other)
    
```

You're welcome to go through the code in detail (there isn't much of it), but it's also a bit dense and it isn't important that you understand the detail here so much as you understand how to use it.  Let's see how to use this code to define a factor over two variables $X_1$ and $X_2$.


```python
f1 = Factor("X1", "X2", {(0,0):0.1, (0,1):0.2, (1,0):0.3, (1,1):0.2, (2,0):0.1, (2,1):0.1})
print(f1)
```

<pre>
Factor(X1, X2) = {
  X1 = 0, X2 = 0: 0.1
  X1 = 0, X2 = 1: 0.2
  X1 = 1, X2 = 0: 0.3
  X1 = 1, X2 = 1: 0.2
  X1 = 2, X2 = 0: 0.1
  X1 = 2, X2 = 1: 0.1
}
</pre>

If we have another factor over the variables $X_2$ and $X_3$, we can multiply these together how you would expect, and there result is the product of the two factors.


```python
f2 = Factor("X2", "X3", {(0,0):0.4, (0,1):0.3, (1,0):0.1, (1,1):0.2})
print(f1*f2)
```

<pre>
Factor(X1, X2, X3) = {
  X1 = 0, X2 = 0, X3 = 0: 0.04000000000000001
  X1 = 0, X2 = 0, X3 = 1: 0.03
  X1 = 0, X2 = 1, X3 = 0: 0.020000000000000004
  X1 = 0, X2 = 1, X3 = 1: 0.04000000000000001
  X1 = 1, X2 = 0, X3 = 0: 0.12
  X1 = 1, X2 = 0, X3 = 1: 0.09
  X1 = 1, X2 = 1, X3 = 0: 0.020000000000000004
  X1 = 1, X2 = 1, X3 = 1: 0.04000000000000001
  X1 = 2, X2 = 0, X3 = 0: 0.04000000000000001
  X1 = 2, X2 = 0, X3 = 1: 0.03
  X1 = 2, X2 = 1, X3 = 0: 0.010000000000000002
  X1 = 2, X2 = 1, X3 = 1: 0.020000000000000004
}
</pre>

We'll use this code to illustrate some of the basic concepts in probability, and hopefully it makes some of the formula more concrete in practice.

