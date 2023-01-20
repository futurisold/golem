<h1 align=center> <b>golem</b> </h1>

<p align="center">
<img src="./golem.png" style="width:75%;height:75%;">
</p>

`Golem` is a minimalistic framework for performing Bayesian analysis. It was heavily inspired by Richard McElreath's [Statistical Rethinking](https://github.com/rmcelreath/stat_rethinking_2023) course, which is a lovely journey into the scientific thinking. It's built on top of PyTorch, and more specifically, it uses the `torch.distributions` module and the available optimization routines.

My main goal is to have a simple interface to sketch ideas and test them quickly. Caution is advise as the tool should be mainly used didactically to get a better grasp of the Bayesian paradigm.

## Installation

Install [PyTorch](https://pytorch.org/get-started/locally/) and then run:

```bash
pip install -e .
```

## Usage

Let's assume we want to model the data collected by Howell between August 1967 and May 1969 in Botswana of the !Kung San tribe (read more [here](https://tspace.library.utoronto.ca/handle/1807/10395)).

```python
import pandas as pd
from golem import Prior, LinearCombination, Model, Golem, MetropolisHastings

# Load the data
data = pd.read_excel('AdultHtWtOne_each_17985.xls')[['htcms', 'wtkgs']].dropna()
H = torch.tensor(data['htcms'].values)
W = torch.tensor(data['wtkgs'].values)

# Define the priors
intercept_prior = Prior('intercept', torch.distributions.Normal, loc=178, scale=20)
slope_prior     = Prior('slope', torch.distributions.LogNormal, loc=0, scale=1)
sigma_prior     = Prior('sigma', torch.distributions.Uniform, low=0, high=50)
mu_prior        = LinearCombination(H, intercept_prior, slope_prior)

# Define the Model
model = Model(W, torch.distributions.Normal, loc=mu_prior, scale=sigma_prior)

# Conjure the Golem
golem = Golem([model])

# Before sampling, we need to estimate the MAP to initialize the sampler
golem.maximum_a_posteriori()

# Sample from the posterior
samples = golem.sample(n_samples=10000, burn_in=5000)
```


