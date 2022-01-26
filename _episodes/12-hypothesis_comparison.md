---
title: >-
   Likelihood ratio: model comparison and confidence intervals
teaching: 40
exercises: 20
questions:
- "How do we fit multi-parameter models to data?"
objectives:
- ""
keypoints:
- ""
---

<script src="../code/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In this episode we will be using numpy, as well as matplotlib's plotting library. Scipy contains an extensive range of distributions in its 'scipy.stats' module, so we will also need to import it and we will also make use of scipy's `scipy.optimize` module. Remember: scipy modules should be installed separately as required - they cannot be called if only scipy is imported.
~~~
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.optimize as spopt
import scipy.integrate as spint
~~~
{: .language-python}


## Which is the best hypothesis?

Imagine that you have two hypotheses, a null hypothesis $$H_{0}$$ and an alternative hypothesis $$H_{A}$$, which you will accept as the alternative to the null. You want to test which is the best hypothesis to explain your data $$D$$.  You might think that your favoured hypothesis should be the one with the greatest [_posterior probability_]({{ page.root }}/reference/#posterior), i.e. you will accept the alternative if $$P(H_{A}\vert D)/P(H_{0}\vert D)>1)$$. 

However, consider the case shown in the plots below, where for simplicity we assume that the data we are using to choose between our hypotheses consists of a single value $$x$$, which may represent a single measurement or even a [_test statistic_]({{ page.root }}/reference/#test-statistic) calculated from multiple measurements. Let's further assume that we have the following posterior probability distributions for each hypothesis as a function of $$x$$, along with their ratio:

<p align='center'>
<img alt="Hypothesis testing: alpha and beta" src="../fig/ep14_hyptesting_alphabeta.png" width="600"/>
</p>

Now, in order to conduct our test we need to place a threshold on the region of $$x$$ values where we will reject $$H_{0}$$ and accept $$H_{A}$$. Based on what we have learned about significance testing so far, we may decide to reject the null hypothesis for values of $$x$$ exceeding some threshold value, i.e. $$x > x_{\rm thr}$$. Taking our alternative hypothesis into account, in the case illustrated above, we might somewhat arbitrarily set $$x_{\rm thr}=8.55$$ because at that point the alternative hypothesis has a three times higher posterior probability than the null hypothesis. However, this approach has some serious problems:

1. If the null hypothesis is true, there is a 12.8% probability (corresponding to the [_significance level_]({{ page.root }}/reference/#significance-level) $$\alpha$$) that it will be rejected in favour of the alternative. This rejection of a true null hypothesis in favour of a false alternative is called a [__false positive__]({{ page.root }}/reference/#false-positive), also known as a [_type I error_].
2. If the null hypothesis is false and the alternative is true, there is a 7.4% probability (corresponding to the integral $$\beta$$) that the alternative will be rejected and the false null will be accepted. The acceptance of a false null hypothesis is known as a [__false negative__]({{ page.root }}/reference/#false-negative) or a [_type  II error_].
3. Furthermore, in this particular case, values of $$x>12.7$$ are more likely to be produced by the null hypothesis than the alternative, according to their posterior probability ratios!

When we carry out significance tests of a null hypothesis, we often place quite a strong requirement on the significance needed to reject the null, because the null is generally chosen as being the simplest and most plausible explanation in the absence of compelling evidence otherwise. The same principle holds for whether we should reject the null hypothesis in favour of an alternative or not. Clearly the possibility of a false negative in our example is too high to be acceptable for a relaible statistical test. We need a better approach to calculate what range of $$x$$ we should use as our threshold for rejecting the null and accepting the alternative.

## The likelihood ratio test

We would like to control the rate of false positive and false negative errors that arise from our test to compare hypotheses. To control the fraction of false negative tests, we should first be able to pre-specify our desired value of $$\alpha$$, i.e. we will only reject the null hypothesis in favour of the alternative if the test gives a $$p$$-value, $$p < \alpha$$ where $$\alpha$$ is set to be small enough to be unlikely. The choice of $$\alpha$$ should also reflect the importance of the outcome of rejecting the null hypothesis, e.g. does this correspond to the detection of a new particle (usually $$5\sigma$$, or just the detection of an astronomical source where one is already known in another waveband (perhaps 3$$\sigma$$)? If $$\alpha$$ is sufficiently small, the risk of a false negative (e.g. detecting a particle or source which isn't real) is low, by definition.

To control the false negative rate, we need to minimise $$\beta$$, which corresponds to the probability that we reject a true alternative and accept the false null hypothesis. The statistical [__power__]({{ page.root }}/reference/#power) of the test is $$1-\beta$$, i.e. by minimising the risk of a false negative we maximise the power of the test.

> ## Significance and power
> Consider a test where rejection of the null hypothesis ($$H_{0}$$) (and acceptance of the alternative $$H_{A}$$) occurs when the value of the test statistic $$x$$ lies in a rejection region $$R$$. We then define:
>
> [__Significance__]({{ page.root }}/reference/#significance-level): 
>
> $$\int_{R} P(H_{0}\vert x)\mathrm{d}x = \alpha$$
>
> [__Power__]({{ page.root }}/reference/#power): 
>
> $$\int_{R} P(H_{A}\vert x)\mathrm{d}x = 1-\beta$$
>
{: .callout}

Given a desired significance level $$\alpha$$, what is the rejection region that maximises the statistical power of a test? The answer to this question is given by the [__Neyman-Pearson Lemma__]({{ page.root }}/reference/#neyman-pearson-lemma), which states that the rejection region that maximises statistical power is given by all $$x$$ that have a large enough [__likelihood ratio__]({{ page.root }}/reference/#likelihood-ratio):

$$\frac{P(H_{A}\vert x)}{P(H_{0}\vert x)} > c$$

where c if fixed such that the test has the desired significance:

$$\int_{R} P(H_{0}\vert x) \mathrm{Hv}\left[\frac{P(H_{A}\vert x)}{P(H_{0}\vert x)}-c\right] \mathrm{d}x = \alpha$$

where $$\mathrm{Hv}[y]$$ is the Heaviside step function, which is zero for negative $$y$$ and 1 otherwise.

To see how this works, we will take our example above and require a fairly minimal significance level $$\alpha=0.05$$.

<p align='center'>
<img alt="Hypothesis testing: Neyman-Pearson" src="../fig/ep14_hyptesting_np.png" width="600"/>
</p>

The figure shows two regions (shaded red and blue) where the false positive rate is 5%. The false negative rate is given by the area of the alternative hypothesis curve $$P(H_{A}\vert x)$$ _outside_ this region. For the blue shaded region, bounded only to the left at $$x_{\mathrm{thr}}$$ with the dashed line as for a standard significance test, the false negative rate is 86%. For the Neyman-Pearson red shaded region, bounded on either side by the dotted lines at $$x_{1}$$ and $$x_{2}$$, the false negative rate is 36%. This means that the Neyman-Pearson likelihood ratio test is substantially more powerful: if the alternative hypothesis is correct it will be favoured versus the null 64% of the time, while using the standard significance threshold $$x_{\mathrm{thr}}$$ will only lead to the correct alternative being favoured 14% of the time, a factor $$>$$4.5 times worse!


## Practical application to model fitting: Wilks' theorem

In general it will be very challenging to calculate the critical threshold to maximise the power of a likelihood ratio test, since one must integrate over the posterior distributions for both the null and alternative hypotheses. However, with some simplifying assumptions that commonly apply in model-fitting, there is an easy way to carry out the test, by applying [__Wilks' theorem__]({{ page.root }}/reference/#wilks-theorem).

First of all, assume that the null hypothesis is a model with $$n$$ parameters $$\theta_{i}$$, with likelihood:

$$p(\mathbf{x} \vert H) = p(\mathbf{x}\vert \theta_{1}, \theta_{2}, ... , \theta_{n})$$

where $$\mathbf{x}$$ is the vector of data points. Now further assume that of the $$n$$ model parameters, a number $$k$$ of them are [__free parameters__]({{ page.root }}/reference/#free-parameter), which are free to vary to find their best-fitting [__MLEs__]({{ page.root }}/reference/#mle). The remaining $$m=n-k$$ parameters are fixed and unable to vary when fitting the model (in model-fitting terminology we say that the parameters are _frozen_).

Now assume that there is an alternative hypothesis with likelihood $$p(\mathbf{x} \vert A)$$, which is the same overall model but in which the $$m$$ fixed parameters are now allowed to vary (they are freed or in some model-fitting software, 'thawed'). The null hypothesis is said to be a nested model of the alternative, which in geometric terms means that its likelihood surface (likelihood vs. parameters) is a _sub-manifold_ of a more multi-dimensional hyper-surface describing the likelihood over the parameter space of the alternative hypothesis.

Now let's look at the likelihood ratio between the null and the alternative hypotheses. We can define the [__log-likelihood-ratio__]({{ page.root }}/reference/#log---likelihood-ratio):

$$\Lambda = 2 \ln \frac{p(\mathbf{x}\vert A)}{p(\mathbf{x}\vert H)}$$

The threshold $$c$$ for rejecting the null hypothesis is obtained from:

$$\int^{\infty}_{c} p(\Lambda\vert H)\mathrm{d}\Lambda = \alpha$$

which is equivalent to asking the question: __assuming the null hypothesis is correct, what is the chance that I would see this value of the log-likelihood-ratio (or equivalently: this difference between the alternative and the null log-likelihoods).__

__Wilks' theorem__ states that in the large-sample limit (i.e. sufficient data), or equivalently, when the MLEs of the additional free parameters in the alternative hypothesis are normally distributed:

$$p(\Lambda\vert H) \sim \chi^{2}_{m}$$

i.e. the log-likelihood-ratio is distributed as a $$\chi^{2}$$ distribution with degrees of freedom equal to the number of extra free parameters in the alternative hypothesis compared to the null hypothesis. That means we can look at the difference in weighted-least squares (or log-likelihood) statistics  between the null and alternative best fits and it will tell us whether the null is rejected or not, by calculating the significance level ($$p$$-value) for $$\Delta \chi^{2}$$ or $$-2\Delta L$$, with respect to a $$\chi^{2}_{m}$$ distribution.

To illustrate how to use likelihood ratio tests in this way, we will show a couple of examples.


## Adding model components: is that a significant emission line?

Looking at the spectral plot of our $$\gamma$$-ray photon data from the previous episode, there is a hint of a possible feature at around 70 GeV.  Could there be an emission feature there?  This is an example of a hypothesis test with a nested model where the main model is a power-law plus a Gaussian emission feature.  The simple power-law model is nested in the power-law plus Gaussian: it is this model with a single constraint, namely that the Gaussian flux is zero, which is our null hypothesis (i.e. which should have a lower probability than the alternative with more free parameters).  Since zero gives a lower bound of the parameter for an emission feature (which is not formally allowed for likelihood ratio tests), we should be sure to allow the Gaussian flux also to go negative (to approximate an absorption feature). 

Before we start, we should first copy into our notebook and run the following functions which we defined in the previous episode: `histrebin`, `pl_model`, `model_int_cf` and `plot_spec_model`. Then we'll run the code below to load in the data, fit it with a simple power-law model and plot the data, model and data/model ratio:

~~~
#  First read in the data.  This is a simple (single-column) list of energies:
photens = np.genfromtxt('photon_energies.txt')

# Now we make our unbinned histogram.  We can keep the initial number of bins relatively large.
emin, emax = 10., 200.   # We should always use the known values that the data are sampled over 
                         # for the range used for the bins!
nbins = 50
counts, edges = np.histogram(photens, bins=nbins, range=[emin,emax], density=False)

# And now we use our new function to rebin so there are at least mincounts counts per bin:
mincounts = 20  # Here we set it to our minimum requirement of 20, but in principle you could set it higher
counts2, edges2 = histrebin(mincounts,counts,edges)

bwidths = np.diff(edges2) # calculates the width of each bin
cdens = counts2/bwidths # determines the count densities
cdens_err = np.sqrt(counts2)/bwidths # calculate the errors: remember the error is based on the counts, 
# not the count density, so we have to also apply the same normalisation.
energies = (edges2[:-1]+edges2[1:])/2.  # This calculates the energy bin centres

model = pl_model
p0 = [2500.0, -1.5]  # Initial power-law parameters
ml_cfpars, ml_cfcovar = spopt.curve_fit(lambda energies, *parm: model_int_cf(energies, edges2, model, *parm),
                                        energies, cdens, p0, sigma=cdens_err)
err = np.sqrt(np.diag(ml_cfcovar))
print("Covariance matrix:",ml_cfcovar)

print("Normalisation at 1 GeV = " + str(ml_cfpars[0]) + " +/- " + str(err[0]))
print("Power-law index = " + str(ml_cfpars[1]) + " +/- " + str(err[1]))
minchisq = np.sum(((cdens - model_int_cf(energies, edges2, model, *ml_cfpars))/cdens_err)**2.)

print("Minimum Chi-squared = " + str(minchisq) + " for " + str(len(cdens)-len(p0)) + " d.o.f.")
print("The goodness of fit is: " + str(sps.chi2.sf(minchisq,df=(len(cdens)-len(p0)))))

# Return the model y-values for model with parameters equal to the MLEs
best_model = model_int_cf(energies, edges2, model, *ml_cfpars)
# Now plot the data and model and residuals
plot_spec_model(edges2,cdens,cdens_err,best_model)
~~~
{: .language-python}
~~~
Covariance matrix: [[ 1.38557897e+05 -1.43587826e+01]
 [-1.43587826e+01  1.57030675e-03]]
Normalisation at 1 GeV = 2695.8605004131714 +/- 372.23365894181904
Power-law index = -1.5725093885531372 +/- 0.0396270962109368
Minimum Chi-squared = 20.652085891889353 for 20 d.o.f.
The goodness of fit is: 0.41785865938156186
~~~
{: .output}

<p align='center'>
<img alt="Power-law photon histogram" src="../fig/ep11_pldatamodel.png" width="500"/>
</p>

The model is formally a good fit, but there may be an addition to the model which will formally give an even better fit. In particular, it looks as if there may be an emission feature around 70~GeV. Will adding a Gaussian emission line here make a significant improvement to the fit? To test this, we must use the (log)-likelihood ratio test. 

First, let's define a power-law plus a Gaussian component.

~~~
def plgauss_model(x, parm):
    '''Power-law plus Gaussian function.
       Inputs:
           x - input x value(s) (can be list or single value).
           parm - parameters, list of PL normalisation (at x = 1) and power-law index, Gaussian mean,
                  Gaussian sigma and Gaussian normalisation.'''
    pl_norm = parm[0]  # here the function given means that the normalisation corresponds to that at a value 1.0
    pl_index = parm[1]
    gmu = parm[2]
    gsig = parm[3]
    gnorm = parm[4]
    # The line is a Gaussian shape with a normalisation equal to the number of counts in the line
    gflux = np.exp(-0.5*((x - gmu)/gsig)**2)/(gsig*np.sqrt(2.*np.pi))
    return pl_norm*x**pl_index + gnorm*gflux
~~~
{: .language-python}

Now initialise the fit parameters and set up the model name. Since the possible emission feature is quite narrow, you need to be fairly precise about the starting energy of the Gaussian component, otherwise the fit will not find the emission feature:

~~~
p0 = [2000.0, -1.6, 70.0, 10.0, 0.0]
model = plgauss_model
~~~
{: .language-python}

From here you can run the same commands as before. You may get a warning about roundoff error in the integration, which causes problems with the calculation of covariance values, but the fit itself is fine and shows improvement in the chi-squared and goodness-of-fit:

~~~
Normalisation at 1 GeV = 2795.9371981847416 +/- 355.8039432852781
Power-law index = -1.5861450028136679 +/- 0.036851576689611615
Line energy = 70.80116290616805 +/- 11124267.195525434
Line width (sigma) = 0.2766707478158328 +/- 5.999748843796642
Line normalisation = 16.238525328480886 +/- 5.999748843796642
Minimum Chi-squared = 14.430153676446889 for 17 d.o.f.
The goodness of fit is: 0.6364535780057641
~~~
{: .output}

<p align='center'>
<img alt="Power-law plus Gaussian photon histogram" src="../fig/ep12_plgaussdatamodel.png" width="500"/>
</p>

Now we can assess the significance of the improvement using Wilks' theorem. We can see that the fit has improved by allowing the line flux to be non-zero, with the chi-squared dropping from 20.65 to 14.43, i.e. $$\Delta \chi^{2}$$, measured from subtracting the worse (higher) value from the better one, is 6.22.  Is this a significant improvement?

One important question here is: what is the number of additional free parameters?  Wilks' theorem tells us that the $$\Delta \chi^{2}$$ in going from a less constrained model to a more constrained one is itself distributed as $$\chi^{2}_{m}$$ where $$m$$ is the number of additional constraints in the more constrained model (or equivalently, the number of additional free parameters in the less constrained model).  In our case, it seems like $$m=3$$, but for an emission line we should be careful: the line energy is not really a 'nestable' model parameter because the likelihood does not smoothly change if we change the position of such a sharp feature.  The line width might be considered as a parameter, but often is limited by the resolution of the instrument which applies a significant lower bound, also making the likelihood ratio approach unsuitable.  Therefore for simplicity here it is better to do the test assuming only the flux as the additional constraint, i.e. the null hypothesis is for flux = 0.  

Thus, we have a $$\chi^{2}_{1}$$ distribution and we can estimate the significance of our improvement using `print("p-value for our delta-chi-squared: ",sps.chi2.sf(6.22,df=1))` which gives:

~~~
p-value for our delta-chi-squared:  0.01263151128225959
~~~
{: .output}

> ## The Bonferroni correction
> The significance for the possible emission line above is not very impressive! Furthermore, we haven't allowed for the fact that in this case we did not expect to find a line at this energy ___a priori___ - we only saw there was a possible feature there ___a posteriori___, i.e. 'after the fact'. A crude way to correct for this is to allow for the number of 'hidden' trials that we effectively conducted by searching across the residuals by eye for a line-like feature. Then we ask the question 'if I carried out $$n$$ trials, what is the chance I would find the observed $$p$$-value by chance in at least one trial?'. In probability terms, this is the _complement_ of the question 'what is the chance that I would see no $$p$$-value of this magnitude or smaller in $$n$$ trials?'. To answer this question, we just consider a Binomial probability with $$\theta=p$$, $$n=20$$ trials and $$x=0$$ successes:
>
> $$p(0|n,\theta) = (1-\theta)^{n} = (1-0.01314)^{20} \simeq 0.77 $$
>
> Here we have estimated the effective number of trials, which we take to be the ratio of the logarithmic range of energies observed divided by the approximate logarithmic energy width of the possible line feature, which spans a couple of bins. The chance that we would see at least 1 success ($$p< 0.01314$$) in the resulting 20 trials is then 0.23, i.e. much less significant than our already not-impressive $$p=0.013$$.
>
> Thus we conclude that the apparent emission feature at ~70 GeV is not significant. Note that if we were even more conservative and assumed $$k=2$$ or 3 for our constraints, our $$p$$-value would be even larger and therefore even less significant. 
>
> The correction for the number of trials described above is known as a ___Bonferroni correction___ - the estimate of the number of trials seems a bit hand-wavy here, based on a heuristic which is somewhat subjective. In Bayesian terms, we can deal with this question (and its subjectivity) more rigorously, including the use of the prior (which represents our uncertainty over where any line can be located). The Bayesian approach is beyond the scope of this course, but an introduction can be found in Chapter 4 of Sivia's book.
>
{: .callout}

## Relation to errors and limits: upper limit on an emission line normalisation

Confidence intervals and upper or lower limits can be thought of as the model calculated for a fixed parameter value, such that the best-fit becomes significantly improved vs. the constrained fit at the confidence interval or limit boundary. In this way, we can interpret confidence intervals and upper and lower limits in the language of hypothesis testing and think of the model evaluated at the interval/limit bounds as the contrained null hypothesis with $$m$$ fewer free parameters than the alternative best-fitting model, where $$m$$ is the dimensionality required for the confidence region. Thus we obtain our result from the previous episode, of using the change in $$\chi^{2}$$ statistic or $$L$$ to define confidence intervals/regions.

Let's imagine that our $$\gamma$$-ray photon data arises from an extreme cosmic explosion like a gamma-ray burst, and that a model predicts that we should see a Gaussian emission line appearing in our spectrum at $$E_{\rm line}=33.1$$ GeV, with Gaussian width $$\sigma=5$$ GeV. There doesn't appear to be feature there in our spectrum, but the line flux (i.e. normalisation) is not specified by the model, so it is an adjustable parameter which our data can constrain. What is the 3-$$\sigma$$ upper limit on the line flux?

For this task we can repurpose the functions `grid1d_chisqmin` and `calc_error_chisq`, defined in the previous episode in order to calculate exact confidence intervals from the $$\chi^{2}$$ statistics calculated for a grid of parameter values. For 1 constrained parameter (i.e. 1 degree of freedom) and a 3-$$\sigma$$ upper limit, we need to find when $$\chi^{2}$$ has changed (relative to the $$\chi^{2}$$ for zero line flux) by $$\Delta \chi^{2}=3^{2}=9$$. This was an easy calculation by hand, but with `scipy.stats` distributions you can also use the _inverse survival function_ which is the survival-function equivalent of the ppf (which is the inverse cdf), e.g. try:
~~~
print(sps.chi2.isf(2*sps.norm.sf(3),df=1))
~~~
{: .language-python}

Now the modified versions of the functions. The `calc_error_chisq` function is modified to calculate only the upper value of an interval, consistent with an upper limit.

~~~
def grid1d_chisqmin_cfint(a_index,a_range,a_steps,parm,model,xval,yval,dy,xedges):
    '''Finds best the fit and then carries out chisq minimisation for a 1D grid of fixed parameters.
       Input: 
            a_index - index of 'a' parameter (in input list parm) to use for grid.
            a_range, a_steps - range (tuple or list) and number of steps for grid.
            parm - parameter list for model to be fitted.
            model - name of model function to be fitted.
            xval, dyval, dy - data x, y and y-error arrays
        Output: 
            a_best - best-fitting value for 'a'
            minchisq - minimum chi-squared (for a_best)
            a_grid - grid of 'a' values used to obtain fits
            chisq_grid - grid of chi-squared values corresponding to a_grid'''
    a_grid = np.linspace(a_range[0],a_range[1],a_steps)
    chisq_grid = np.zeros(len(a_grid))
    # First obtain best-fitting value for 'a' and corresponding chi-squared
    ml_cfpars, ml_cfcovar = spopt.curve_fit(lambda xval, *parm: model_int_cf(xval, xedges, 
                            model, parm), xval, yval, parm, sigma=dy)
    
    minchisq = np.sum(((yval-model_int_cf(xval,xedges,model,*ml_cfpars))/dy)**2)
    a_best = ml_cfpars[a_index]
    # Now remove 'a' from the input parameter list, so this parameter may be frozen at the 
    # grid value for each fit
    free_parm = np.delete(parm,a_index)
    # Now fit for each 'a' in the grid, to do so we must use a lambda function to insert the fixed 
    # 'a' into the model function when it is called by curve_fit, so that curve_fit does not use 
    # 'a' as one of the free parameters so it remains at the fixed grid value in the fit.
    for i, a_val in enumerate(a_grid):        
        ml_cfpars, ml_cfcovar = spopt.curve_fit(lambda xval, *parm: model_int_cf(xval, xedges, 
                            model, parm), xval, yval, parm, sigma=dy)        
        chisq_grid[i] = np.sum(((yval-model_int_cf(xval,xedges,model,
                                                   *np.insert(ml_cfpars,a_index,a_val)))/dy)**2)
        print(i+1,'steps: chisq =',chisq_grid[i],'for a =',a_val,' minimum = ',minchisq,' for a =',a_best)
    return a_best, minchisq, a_grid, chisq_grid  
    
def calc_upper_chisq(delchisq,minchisq,a_grid,chisq_grid):
    '''Function to return upper values of a parameter 'a' for a given delta-chi-squared
       Input:
           delchisq - the delta-chi-squared for the confidence interval required (e.g. 1 for 1-sigma error)
           a_grid, chisq_grid - grid of 'a' and corresponding chi-squared values used for interpolation'''
    # First interpolate over the grid for values > a_best and find upper interval bound
    chisq_interp_upper = spinterp.interp1d(chisq_grid,a_grid)
    a_upper = chisq_interp_upper(minchisq+delchisq)
    return a_upper
~~~
{: .language-python}

Now we can run the grid search to find the upper limit. However, we need to bear in mind that we must keep the Gaussian feature mean energy and width $$\sigma$$ fixed at the states values for these fits! `curve_fit` will vary all the parameters given to it, so to fix ('freeze') parameters, we should do this by creating a bespoke model function for `curve_fit` to use, which makes use of our original model but freezes and 'hides' the frozen parameters from `curve_fit`:

~~~
# Give the fixed parameters and define a new function to hide them from curve_fit:
fixed_en = 33.1
fixed_sig = 5.0
new_model = lambda x, parm: plgauss_model(x,np.insert(np.insert(parm,2,fixed_en),3,fixed_sig))

# Below is the parameter list which curve_fit will see initially, with PL norm, 
# PL index and Gaussian line normalisation.
parm = [2000.0, -1.6, 0.0]
# Now we select the parameter to step through and give the range and number of steps:
a_index = 2
par_range = [0,40]
n_steps = 100
# Run the grid calculation
a_best, minchisq, a_grid, chisq_grid = grid1d_chisqmin_cfint(a_index,par_range,n_steps,parm,new_model,
                             energies,cdens,cdens_err,edges2)

# Now give the output
delchisq = 9
a_upper = calc_upper_chisq(delchisq,minchisq,a_grid,chisq_grid)
print("3-sigma upper limit on line flux: ", a_upper)
~~~
{: .language-python}

Note that for a meaningful 3-$$\sigma$$ upper limit according to Wilks' theorem, we must compare with the best-fit (the alternative hypothesis) even if the best-fitting line flux is non-zero, and since it is an upper limit and not an interval, we just state the absolute value of the flux not the difference from the best-fitting value.

We can also plot our grid to check that everything has worked okay and that there is a smooth variation of the $$\Delta \chi^{2}$$ with the line flux. We also show the location for $$\Delta \chi^{2}$$ on the plot (the corresponding line flux is $$27.4$$ counts).

<p align='center'>
<img alt="Power-law photon histogram" src="../fig/ep12_upperlim.png" width="500"/>
</p>


> ## Programming challenge: constraining spectral features
> 
> A particle accelerator experiment gives (after cleaning the data of known particle events) a set of measured event energies (in GeV) contained in the file [`event_energies.txt`][event_data]. The detector detects events with energies in the range 20 to 300 GeV and the background events produce a continuum spectrum which follows an exponentially cut-off power-law shape:
>
> $$N_{\rm cont}(E)=N_{0}E^{-\Gamma} \exp(-E⁄E_{\rm cut})$$
>
> where $$N_{\rm cont}(E)$$ is the number of continuum photons per GeV at energy $$E$$ GeV. The normalisation $$N_{0}$$, power-law index $$\Gamma$$ and cut-off energy $$E_{\rm cut}$$ (in GeV) are parameters to be determined from the data. Besides the background spectrum, a newly-discovered particle which is under investigation produces an additional feature in the spectrum somewhere between 80-85 GeV, with a Gaussian profile:
>
> $$N_{\rm Gauss}(E)=\frac{N_{\rm total}}{\sigma\sqrt{2\pi}} \exp\left(-(E-E_{\rm cent})^{2}/(2\sigma^{2})\right)$$
> 
> Where $$N_{\rm Gauss}(E)$$ is the number of photons per GeV in the Gaussian feature, at energy $$E$$ GeV. $$N_{\rm total}$$ is a normalisation corresponding to the (expected) total number of photons in the feature, $$E_{\rm cent}$$ is the Gaussian centroid energy and $$\sigma$$ is the Gaussian width (i.e. standard deviation). The $$\sigma$$ of the Gaussian feature can be assumed to be fixed at the instrument resolution, which is 1.3 GeV and is the same at all energies.
> 
> - a) Calculate and plot (with appropriate axes and in appropriate units) a histogram of the event spectrum and then fit the continuum-only model to the data. Use the data/model ratio to identify the possible location(s) of any Gaussian features, including the known feature in the 80-85 GeV range.
> - b) Now add a Gaussian profile to your continuum model to fit the feature in the 80-85 GeV range. Fit the new combined model and determine the continuum and Gaussian MLEs, their errors (use covariance matrix __or__ 1-D grid search) and the model goodness of fit. Confirm that the Gaussian feature makes a significant improvement to the fit compared to the continuum-only model. Estimate what the significance of the line feature would be __if you did not already know its expected energy range__.
> - c) Prof. Petra Biggs has proposed a theory that predicts that a new particle should produce a feature (also a Gaussian at the instrument resolution) at 144.0 GeV. Use your data to set 3-sigma upper limit on the normalization (in terms of expected number of events N_total) that can be produced by this particle.
> 
{: .challenge}


[event_data]: https://github.com/philuttley/statistical-inference/tree/gh-pages/data/event_energies.txt


{% include links.md %}


