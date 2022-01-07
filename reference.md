---
layout: reference
permalink: /reference/
root: ..
---

<script src="../code/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Glossary

{:auto_ids}
accuracy
:   The relative amount of non-random deviation from the 'true' value of a quantity being measured. Measurements of the same quantity but with smaller [systematic error](#systematic-error) are more accurate. 
See also: [precision](#precision)

argument
:   A value given to a python function or program when it runs. In python programming, the term is often used interchangeably (and inconsistently) with 'parameter', but here we restrict the use of the term 'parameter' to probability distributions only.

Bayesian
:   Approach to understanding probability in terms of the belief in how likely something is to happen, including [prior](#prior) information. Mathematically the approach is described by Bayes' theorem, which can be used to convert the calculated [likelihood](#likelihood) of obtaining a particular data set given a hypothesis, into the [posterior](#posterior) probability of the hypothesis being correct given the data.

Bessel's correction
:   The correction $$\frac{1}{n-1}$$ (where $$n$$ is sample size) to the arithmetic sum of sample [variance](#variance) so that it becomes an unbiased [estimator](#estimator) of the population variance. The value of 1 subtracted from sample size is called the __degrees of freedom__. The correction compensates for the fact that the part of population variance that leads to variance of the sample mean is already removed from the sample variance, because it is calculated w.r.t. to sample mean and not population mean.

bias
:   The bias of an [estimator](#estimator) is the difference between the [expected value](#expectation) of the estimator and the true value of the quantity being estimated.

bivariate
:  Involving two [variates](#random-variate), e.g. __bivariate data__ is a type of data consisting of observations/measurements of two different variables; __bivariate analysis__ studies the relationship between two paired measured variables.

categorical data
:   A type of data which takes on non-numerical values (e.g. subatomic particle types).

Cauchy-Schwarz inequality
:   TBD

central limit theorem
: The theorem states that under general conditions of finite mean and variance, sums of variates drawn from non-normal distributions will tend towards being normally distributed, asymptotically with sample size $$n$$.

cdf
:   A cumulative distribution function (cdf) gives the cumulative probability that a random variable following a given [probability distribution](#probability-distribution) may be less than or equal to a given value, i.e. the cdf gives $$P(X \leq x)$$. The cdf is therefore limited to have values over the interval $$[0,1]$$. For a continuous random variable, the derivative function of a cdf is the [pdf](#pdf). For a discrete random variable, the cdf is the cumulative sum of the [pmf](#pmf).

chi-squared fitting
:   See [weighted least squares](#weighted-least-squares)

chi-squared test
:   TBD

conditional probability
:   If the probability of an event $$A$$ depends on the occurence of another event $$B$$, $$A$$ is said to be conditional on $$B$$. The probability of $$A$$ happening if $$B$$ also happens is denoted $$P(A\vert B)$$, i.e. the probability of '$$A$$ conditional on $$B$$' or of '$$A$$ given $$B$$'. See also: [independence](#independence).

confidence interval
:   TBD

confidence level
:   Often used as an alternative form when stating the [significance level](#significance-level) ($$\alpha$$), it is expressed as 1 minus the significance when quoted as a percentage. E.g. _'the hypothesis is ruled out at the 95% confidence level'_ (for $$\alpha=0.05$$).

continuous
:   Relating to a continuous [random variable](#random-variable), i.e. may take on a continuous and infinite number of possible values within a range specified by the corresponding continuous [probability distribution](#probability-distribution).

correlated variables
:   TBD

correlation coefficient
:   TBD

covariance
:   TBD

covariance matrix
:   TBD

discrete
:   Relating to a discrete [random variable](#random-variable), i.e. may only take on discrete values (e.g. integers) within a range specified by the corresponding discrete [probability distribution](#probability-distribution).

distributions - Bernoulli
:   The result of single trial with two outcomes (usually described as 'success' and 'failure', with probability of success $$\theta$$) follows a Bernoulli distribution. If success is defined as $$X=1$$ then a variate distributed as $$X\sim \mathrm{Bern}(\theta)$$ has [pmf](#pmf) $$p(x\vert \theta) = \theta^{x}(1-\theta)^{1-x} \quad \mbox{for }x=0,1$$, with $$E[X]=\theta$$ and $$V[X]=\theta(1-\theta)$$. See also: [binomial distribution](#distributions---binomial).

distributions - binomial
:   The distribution of number of 'successes' $$x$$ produced by $$n$$ repeated [Bernoulli](#distributions---Bernoulli) trials with probability of success $$\theta$$. $$X\sim \mathrm{Binom}(n,\theta)$$ has [pmf](#pmf) $$p(x\vert n,\theta) = \frac{n!}{(n-x)!x!} \theta^{x}(1-\theta)^{n-x} \quad \mbox{for }x=0,1,2,...,n.$$, with $$E[X]=n\theta$$ and $$V[X]=n\theta(1-\theta)$$.

distributions - chi-squared
:   TBD

distributions - lognormal
:   TBD

distributions - normal
:   A normally distributed variate $$X\sim N(\mu,\sigma)$$ has pdf $$p(x\vert \mu,\sigma)=\frac{1}{\sigma \sqrt{2\pi}} e^{-(x-\mu)^{2}/(2\sigma^{2})}$$, and [location parameter](#parameter) $$\mu$$, [scale parameter](#parameter) $$\sigma$$. Mean $$E[X]=\mu$$ and variance $$V[X]=\sigma^{2}.$$ The limiting distribution of sums of random variables (see: [central limit theorem](#central-limit-theorem)). The __standard normal distribution__ has $$\mu=0$$, $$\sigma^{2}=1$$.

distributions - Poisson
:   The Poisson distribution gives the probability distribution of counts measured in a fixed interval or bin, assuming that the counts are independent follow a constant mean rate of counts/interval $$\lambda$$. For variates $$X\sim \mathrm{Pois}(\lambda)$$, the [pmf](#pmf) $$p(x \vert \lambda) = \frac{\lambda^{x}e^{-\lambda}}{x!}$$, with $$E[X] = \lambda$$ and $$V[X] = \lambda$$. The Poisson distribution can be derived as a limiting case of the [binomial](#binomial) distribution, for an infinite number of trials.

distributions - t
:   The distribution followed by the [$$t$$-statistic](#t-statistic), corresponding to the distribution of variates equal to $$T=X/Y$$ where $$X$$ is drawn from a standard normal distribution and $$Y$$ is the square root of a variate drawn from a scaled [chi-squared distribution](#distributions---chi-squared), for a given number of degrees of freedom $$\nu$$.

distributions - uniform
:   $$X\sim U(a,b)$$ has pdf $$p(x\vert a,b)=\mathrm{constant}$$ on interval $$[a,b]$$ (and zero elsewhere), and [location parameter](#parameter) $$a$$, [scale parameter](#parameter) $$\lvert b-a \rvert$$. Mean $$E[X] = (b+a)/2$$ and variance $$V[X] = (b-a)^{2}/12$$. Uniform [random variates](#random-variate) can be used to generate random variates from any other [probability distribution](#probability-distribution) via the [ppf](#ppf) of that distribution.

estimator
: A method for calculating from data an estimate of a given quantity. For example, the [sample mean](#mean) and [variance](#variance) are estimators of the [population mean](#mean) and [variance](#variance). See also: [bias](#bias), [MLE](#MLE).

event
:   In probability theory, an event is an outcome or set of outcomes of a [trial](#trial) to which a probability can be assigned. E.g. an experimental measurement or sample of measurements, a sample of observational data, a dice roll, a 'hand' in a card game, a sequence of computer-generated random numbers or the quantity calculated from them.

evidence
:   TBD

expectation
:  The expectation value of a quantity, which may be a [random variable](#random variable) or a function of a random variable, is (for [continuous](#continuous) random variables) the integral (over the variable) of the quantity weighted by the [pdf](#pdf) of the variable (or for [discrete](#discrete) random variables, the sum weighted by the [pmf](#pmf)). In [frequentist](#frequentism) terms, expectation gives the mean of the random variates (or function of them) in the case of an infinite number of measurements.

false negative
:   TBD

false positive
:   TBD

frequentist
:  Interpretation of probability which defines the probability of an event as the limit of its frequency in many independent [trials](#trial).

goodness of fit
:   TBD

histogram
:   A method of plotting the distribution of data by binning (assigning data values) in discrete bins and plotting either the number of values or _counts_ (sometimes denoted _frequency_) per bin or normalising by the total bin width to give the _count density_, or further normalising the count density by the total number of counts to give a _probability density_ or sometimes just denoted _density_).

hypothesis
:   In statistical terms a hypothesis is a scientific question which is formulated in a way which can be tested using statistical methods. A __null hypothesis__ is a specific kind of baseline hypothesis which is assumed true in order to formulate a statistical test to see whether it is rejected by the data.

hypothesis test
:   A statistical test, the result of which either rejects a (null) [hypothesis](#hypothesis) to a given significance level (this is also called a [significance test](#significance-test)) or gives a probability that an alternative hypothesis is preferred over the null hypothesis, to explain the data.

independence
:   Two events are independent if the outcome of one does not affect the probability of the outcome of the other. Formally, if the events $$A$$ and $$B$$ are independent, $$P(A\vert B)=P(A)$$ and $$P(A \mbox{ and } B)=P(A)P(B)$$. See also [conditional probability](#conditional-probability).

interquartile range
:   The IQR is a form of [confidence interval](#confidence-interval) corresponding to the range of data values from the 25th to the 75th percentile.

joint probability distribution
:   A probability distribution which describes the joint probability of sampling given combinations of variables. Such distributions are commonly known as [bivariate](#bivariate) (for two variables, i.e. the distribution for a combined sample of two different variates) or [multivariate](#multivariate) for more than two variables.

likelihood
:   TBD

likelihood function
:   TBD

likelihood ratio
:   TBD

marginalisation
:   The procedure of removing [conditional](#conditional-probability) terms from a probability distribution by summing over them, e.g. for discrete [event](#event) $$B$$ and multiple [mutually exclusive](#mutual-exclusivity) events $$A_{i}$$, $$P(B)= \sum\limits_{i=1}^{n} P(B\vert A_{i}) P(A_{i})$$. For a [continuous](#continuous) [joint probability distribution](#joint-probability-distribution), marginalisation corresponds to integration over the conditional parameter, e.g. $$p(x) = \int_{-\infty}^{+\infty} p(x,y)dy = \int_{-\infty}^{+\infty} p(x\vert y)p(y)\mathrm{d}y$$.

mean
:   The (sample) mean $$\bar{x}$$ for a quantity $$x_{i}$$ measured from a [sample](#sample) of data is a [statistic](#statistic) calculated as the average of the quantity, i.e. $$\frac{1}{n} \sum\limits_{i=1}^{n} x_{i}$$. For a random variable $$X$$ defined by a [probability distribution](#probability-distribution) with [pdf](#pdf) $$p(x)$$, the ([population](#population) or distribution) mean $$\mu$$ is the expectation value of the variable, $$\mu=E[X]=\int^{+\infty}_{-\infty} xp(x)\mathrm{d}x$$.

median
:   The median for a quantity measured from a [sample](#sample) of data, is a [statistic](#statistic) calculated as the central value of the ordered values of the quantity.  For a random variable defined by a [probability distribution](#probability-distribution), the median corresponds to the value of the 50th percentile of the variable (i.e. with half the total probability below and above the median value).

member
:   A python variable contained within an [object](#object).

method
:   A python function which is tied to a particular python [object](#object). Each of an object's methods typically implements one of the things it can do, or one of the questions it can answer.

MLE
: TBD

mode
:   The mode is the most frequent value in a [sample](#sample) of data. For a random variable defined by a [probability distribution](#probability-distribution), the mode is the value of the variable corresponding to the peak of the [pdf](#pdf).

multivariate 
:   Involving three or more [variates](#random-variate), e.g. __multivariate data__ is a type of data consisting of observations/measurements of three variables; __multivariate analysis__ studies the relationships between three or more variables, to see which are related and how.

mutual exclusivity
:   Two events are mutually exclusive if they cannot both occur, or equivalently the probability of one occurring is conditional on the other __not__ occurring. I.e. events $$A$$ and $$B$$ are mutually exclusive if $$P(A \mbox{ and } B)=0$$ which occurs if $$P(A\vert B)=0$$. For mutually exclusive events, it follows that $$P(A \mbox{ or } B)=P(A)+P(B)$$.

object
:   A collection of conceptually related python variables ([members](#member)) and functions using those variables ([methods](#method)).

ordinal data
:   A type of [categorical data](#categorical-data) which can be given a relative ordering or ranking but where the differences between ranks are not known or explicitly specified by the categories (e.g. stellar spectral types).

parameter
:   [Probability distributions](#probability-distribution) are defined by parameters which are specific to the distribution, but can be classified according to their effects on the distribution. A __location parameter__ determines the location of the distribution on the variable ($$x$$) axis, with changes shifting the distribution on that axis. A __scale parameter__ determines the width of the distribution and stretches or shrinks it along the $$x$$-axis. __Shape parameters__ do something other than shifting/shrinking/stretching the distribution, changing the distribution shape in some way. Some distributions use a __rate parameter__, which is the reciprocal of the scale parameter.

pdf
:   A probability density function (pdf) gives the probability density (i.e. per unit variable) of a [continuous](#continuous) [probability distribution](#probability-distribution), i.e. the values of the pdf give the relative probability or frequency of occurrence of values of a [random variable](#random-variable). The pdf should be normalised so that the definite integral over all possible values is unity. The integral function of the pdf is the [cdf](#cdf).

percentile
:   Value of an ordered variable (which may be data) below which a given percentage of the values fall (exclusive definition - inclusive definition corresponds to 'at or below which') . E.g. 25% of values lie below the data value corresponding to the 25th percentile. For a random variable, the percentile corresponds to the value of the variable below which a given percentage of the probability is contained (i.e. it is the value of the variable corresponding to the inverse of the [cdf](#cdf) - or [ppf](#ppf) for the percentage probability expressed as a decimal fraction).
See also: [quantile](#quantile).

pmf
:   The probability mass function (pmf) is the discrete equivalent of the [pdf](#pdf), corresponding to the probability of drawing a given integer value from a discrete probability distribution. The sum of pmf values for all possible outcomes from a discrete probability distribution should equal unity. 

population
:  The notional population of random variates or objects from which a sample is drawn. A population may have some real equivalent (e.g. an actual population of objects which is being sampled). In the [frequentist](#frequentism) approach to statistics it can also represent the notional infinite set of [trials](#trial) from which a [random variable](#random-variable) is drawn.

posterior
:   TBD

power
:   TBD

ppf
:   A percent point function (ppf) gives the value of a variable as a function of the cumulative probability that it corresponds to, i.e. it is the inverse of the [cdf](#cdf).

precision
:   The relative amount of random deviation in a quantity being measured. Measurements of the same quantity but with smaller [statistical error](#statistical-error) are more precise. 
See also: [accuracy](#accuracy).

prior
:   TBD

probability distribution
:   Distribution giving the relative frequencies of occurence of a [random variable](#random-variable) (or variables, for [bivariate](#bivariate) and [multivariate](#multivariate) distributions).

$$p$$-value
:   A statistical test probability calculated for a given test statistic and assumptions about how the [test statistic](#test-statistic) is distributed (e.g. depending on the [null hypothesis](#hypothesis) and any other assumptions required for the test).

quantile
:   Values which divide a probability distribution (or ordered data set) into equal steps in cumulative probability (or cumulative frequency, for data). Common forms of quantile include [_percentiles_](#percentile), _quartiles_ (corresponding to steps of 25%, known as 1st, 2nd - the median - 3rd and 4th quartile) and _deciles_ (steps of 10%).

random variable
:   A variable which may taken on random values ([variates](#random-variate)) with a range and frequency specified by a probability distribution.

random variate
:   Also known simply as a 'variate', a random variate is an observed outcome of a [random variable](#random-variable), i.e. drawn from a probability distribution of that variable.

realisation
:  An observed outcome of a random process, e.g. it may be a set of [random variates](#random-variate), or the result of an algorithm applied to a set of random variates.

rug plot
:   A method of plotting [univariate data](#univariate) as a set of (usually vertical) marks representing each data value, along an axis (usually the $$x$$-axis). It is usually combined with a [histogram](#histogram) to also show the frequency or probability distribution of the plotted variable.

sample
:  A set of measurements, drawn from an underlying population, either real (e.g. the height distribution of Dutch adults) or notional (the distribution of possible measurements from an experiment with some random measurement error). A sample may also refer to a set of random variates drawn from a probability distribution.

sample space
:   The set of all possible outcomes of an experiment or trial. 

sampling with replacement
:  Sampling with replacement is when the sampling process does not reduce the set of outcomes that is sampled from. E.g. rolling a dice multiple times is sampling (the numbers on the dice) with replacement, while repeatedly randomly drawing different-coloured sweets from a bag, without putting them back, is sampling without replacement.

seed
:   (pseudo-)Random number generators must be 'seeded' using a number, usually an integer, which is usually provided automatically by a system call, but may also be specified by the user. Starting from a given seed, a random number generator will return a fixed sequence of pseudo-random variates, as long as the generating function is called repeatedly without resetting the seed. This behaviour must be forced in Python using e.g.  `np.random.default_rng(331)` for a starting `seed` argument  equal to 331. Otherwise, if no argument is given, a seed is chosen from system information on the computer (this is often the preferred option, unless the same random number sequence is required each time the code is run).

significance level
:   The significance level $$\alpha$$ is a pre-specified level of probability required from a [significance test](#significance-test) in order for a hypothesis to be rejected, i.e. the hypothesis is rejected if the $$p$$-value is less than or equal to $$\alpha$$.

significance test
:   See [hypothesis test](#hypothesis-test)

standard deviation
:   The standard deviation of a sample of data or a random variable with a probability distribution, is equal to the square-root of [variance](#variance) for that quantity. In the context of time-variable quantities it is also often called the __root-mean-squared deviation__ (or just __rms__).

standard error
:   The standard error is the expected [standard deviation](#standard-deviation) on the [sample mean](#mean) (with respect to the 'true' [population mean](#mean)). For $$n$$ measurements drawn from a population with variance $$\sigma^{2}$$, the standard error is $$\sigma_{\bar{x}} = \sigma/\sqrt{n}$$.

stationary process
:   A random process is said to be stationary if it is produced by [random variates](#random-variate) drawn from a [probability distribution](#probability-distribution) which is constant and does not change over time.

statistic
:   A single number calculated by applying a statistical algorithm or function to the values of the items in a [sample](#sample). The [sample mean](#mean), [sample median](#median) and [sample variance](#variance) are all examples of statistics.

statistical error
:   A _random_ error (deviation from the 'true' value(s)) in quantities obtained or derived from data, possibly resulting from random measurement error in the apparatus (or measurer) or due to intrinsic randomness in the quantity being measured (e.g. photon counts) or the sample obtained (i.e. the sample is a random subset of an underlying population, e.g. of stars in a cluster). 
See also: [systematic error](#systematic-error), [precision](#precision).

systematic error
:   An error that is not random but is a systematic shift away from the 'true' value of the measured quantity obtained from data (or a quantity derived from it). E.g. a systematic error may be produced by a fault in the experimental setup or apparatus, or a flaw in the design of a survey so it is biased towards members of the population being sampled with specific properties in a way that cannot be corrected for.
See also [statistical error](#statistical-error), [accuracy](#accuracy)

statistical test
:   A test of whether a given [test statistic](#test-statistic) is consistent with its distribution under a specified hypothesis (and associated assumptions).

survival function
:   A function equal to 1 minus the [cdf](#cdf), i.e. it corresponds to the probability $$P(X\gt x)$$ and is therefore useful for assessing [$$p$$-values](#$$p$$-value) of [test statistics](#test-statistic).

test statistic
:   A [statistic](#statistic) calculated from data for comparison with a known [probability distribution](#probability-distribution) which the test statistic is expected to follow if certain assumptions (including a given hypothesis about the data) are satisfied. 

trial
:   An 'experiment' which results in a [sample](#sample) of (random) data. It may also refer to the process of generating a sample of random variates or quantities calculated from random variates, e.g. in a numerical experiment or simulation of a random process.

t-statistic
:   A [test statistic](#test-statistic) which is defined for a sample with [mean](#mean) $$\bar{x}$$ and [standard deviation](#standard-deviation) $$s_{x}$$ with respect to a [population](#population) of known mean $$\mu$$ as: $$t = (\bar{x}-\mu)/(s_{x}/\sqrt{n})$$. $$t$$ is drawn from a [t-distribution](#distributions---t) if the sample mean is normally distributed (e.g. via the [central limit theorem](#central-limit-theorem) or if the sample is drawn from a population which is itself normally distributed).

t-test
:   Any test where the [test statistic](#test-statistic) follows a $$t$$-distribution under the null hypothesis, such as tests using the [$$t$$-statistic](#t-statistic).

univariate
:   Involving a single [variate](#random-variate), e.g. __univariate data__ is a type of data consisting only of observations/measurements of a single quantity or characteristic; __univariate analysis__ studies statistical properties of a single quantity such as its statistical moments and/or [probability distribution](#probability-distribution).

variance
:   The (sample) variance $$s_{x}^{2}$$ for a quantity $$x_{i}$$ measured from a [sample](#sample) of data, is a [statistic](#statistic) calculated as the average of the squared deviations of the data values from the [sample mean](#mean) (corrected by [Bessel's correction](#Bessel's_correction)), i.e. $$\frac{1}{n-1} \sum\limits_{i=1}^{n} (x_{i}-\bar{x})^{2}$$. For a random variable $$X$$ defined by a [probability distribution](#probability-distribution) with [pdf](#pdf) $$p(x)$$, the ([population](#population) or distribution) variance $$V[X]$$ is the expectation value of the squared difference of the variable from its [mean](#mean) $$\mu$$, $$V[X] = E[(X-\mu)^{2}] = \int^{+\infty}_{-\infty} (x-\mu)^{2}p(x)\mathrm{d}x$$, which is equivalent to the expectation of squares minus the square of expectations of the variable, $$E[X^{2}]-E[X]^{2}$$. 

weighted least squares
:   TBD

Z-statistic
:   A [test statistic](#test-statistic) which is defined for a [sample mean](#mean) $$\bar{x}$$ with respect to a [population](#population) of known mean $$\mu$$ and [variance](#variance) $$\sigma^{2}$$ as: $$Z = (\bar{x}-\mu)/(\sigma/\sqrt{n})$$. $$Z$$ is drawn from a [standard normal distribution](#distributions---normal) if the sample mean is normally distributed (e.g. via the [central limit theorem](#central-limit-theorem) or if the sample is drawn from a population which is itself normally distributed).

Z-test
:   Any test where the [test statistic](#test-statistic) is normally distributed under the null hypothesis, such as tests using the [$$Z$$-statistic](#z---statistic) (although a $$Z$$-test does not have to use the $$Z$$-statistic).











