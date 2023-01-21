from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.distributions import Distribution


class Prior:
    def __init__(self, estimand: str, dist: Distribution, **params: float) -> None:
        self.dist     = dist(**params)
        self.estimand = estimand
        # Initialize the state of the prior
        self.theta    = torch.ones(1, requires_grad=True)


class LinearCombination:
    def __init__(self, data: torch.Tensor, intercept: Prior, slope: Prior, center: bool = True) -> None:
        self.data      = data - data.mean() if center else data
        self.intercept = intercept
        self.slope     = slope

    @property
    def params(self) -> torch.Tensor: return self.intercept.theta + self.slope.theta * self.data


class Model:
    def __init__(self,
                 data: torch.Tensor,
                 model: Distribution,
                 **priors: Prior | LinearCombination) -> None:
        self.data   = data
        self.model  = model
        self.priors = priors
        assert 'log_prob' in dir(self.model), 'The model must have a \"log_prob\" method!'

    def log_prior(self) -> float:
        total = 0
        for prior in self.priors.values():
            if isinstance(prior, Prior): total += prior.dist.log_prob(prior.theta)
            elif isinstance(prior, LinearCombination):
                total += prior.intercept.dist.log_prob(prior.intercept.theta)
                total += prior.slope.dist.log_prob(prior.slope.theta)
            else: raise ValueError('Invalid prior type!')

        return total

    def log_likelihood(self) -> float:
        params = {}
        for estimand, prior in self.priors.items():
            if isinstance(prior, Prior): params |= {estimand: prior.theta}
            elif isinstance(prior, LinearCombination): params |= {estimand: prior.params}
            else: raise ValueError('Invalid prior type!')

        return sum(self.model(**params).log_prob(self.data))

    def log_posterior(self) -> float: return self.log_prior() + self.log_likelihood()

    def iter_params(self) -> dict[str, torch.Tensor]:
        params = {}
        for prior in self.priors.values():
            if isinstance(prior, Prior): params |= {prior.estimand: prior.theta}
            elif isinstance(prior, LinearCombination):
                params |= {prior.intercept.estimand: prior.intercept.theta}
                params |= {prior.slope.estimand: prior.slope.theta}
            else: raise ValueError('Invalid prior type!')

        return params

    def update(self, estimand: str, theta: torch.Tensor) -> None:
        # update available only for priors or combinations of priors
        for prior in self.priors.values():
            if isinstance(prior, Prior):
                if prior.estimand == estimand and prior.dist.support.check(theta): prior.theta = theta
            elif isinstance(prior, LinearCombination):
                if prior.intercept.estimand == estimand and prior.intercept.dist.support.check(theta): prior.intercept.theta = theta
                elif prior.slope.estimand == estimand and prior.slope.dist.support.check(theta): prior.slope.theta = theta
                else: continue # if the estimand is not found, continue to the next prior
            else: raise ValueError('Invalid prior type!')


class Golem:
    def __init__(self,
                 models: list[Model],
                 assumptions: dict[str, str] | None = None) -> None:
        self.models = models
        self.assumptions = assumptions

        # Metroplis-Hastings parameters
        self.__sampler = torch.distributions.Normal(0, 1) # keep it simple and naive - use a normal distribution
        self.__steps = 0
        self.__accepted = 0 # keep track of the accepted steps to tune the scale of the sampler every 100 steps
        self.__tune_interval = 100
        self.__tune_countdown = self.__tune_interval


    def build_dag(self, show=False) -> nx.Graph:
        assert self.assumptions is not None, "No generative assumptions provided!"
        graph = nx.DiGraph()
        for fr, to in self.assumptions.items():
            graph.add_edge(fr, to)

        if show:
            nx.draw(graph, arrows=True, with_labels=True)
            plt.show()

        return graph

    def maximum_a_posteriori(self, lr: float = 1e-2, n_iter: int = 5000):
        params = [theta for model in self.models for theta in model.iter_params().values()]
        optim = torch.optim.AdamW(params, lr=lr, amsgrad=True)
        for _ in range(n_iter):
            optim.zero_grad()
            loss = -sum(model.log_posterior() for model in self.models)
            loss.backward()
            optim.step()

        return [model.iter_params() for model in self.models]

    def sample(self, n_samples: int = 1000, burn_in: int = 100) -> dict[str, torch.Tensor]:
        # burn-in
        for _ in range(burn_in): self.__step()

        # sample
        samples = defaultdict(list)
        while self.__steps < (n_samples + burn_in):
            self.__step()
            for i, model in enumerate(self.models):
                for estimand, param in model.iter_params().items():
                    samples[f"{estimand}_{i}"].append(param.item())

        return samples

    def __step(self) -> None:
        old_state = deepcopy(self.models)
        old_score = sum(model.log_posterior() for model in old_state)

        for model in self.models:
            for estimand, param in model.iter_params().items():
                self.__sampler.loc = param
                proposal = self.__sampler.sample(param.shape)
                model.update(estimand, proposal)

        new_score = sum(model.log_posterior() for model in self.models)

        if not self.__accept_proposal(new_score, old_score): self.models = old_state
        else: self.__accepted += 1

        self.__steps += 1
        self.__tune_countdown -= 1
        if self.__tune_countdown == 0:
            self.__tune()
            self.__tune_countdown = self.__tune_interval

    def __tune(self) -> None:
        # borrowed from PyMC3
        ratio = self.__acceptance_ratio()
        if ratio < 0.001:  self.__sampler.scale *= 0.1 # reduce by 90%
        elif ratio < 0.05: self.__sampler.scale *= 0.5 # reduce by 50%
        elif ratio < 0.20: self.__sampler.scale *= 0.9 # reduce by 10%
        elif ratio > 0.95: self.__sampler.scale *= 10  # increase by 1000%
        elif ratio > 0.75: self.__sampler.scale *= 2   # increase by 100%
        elif ratio > 0.50: self.__sampler.scale *= 1.1 # increase by 10%

    def __accept_proposal(self, new_score: float, old_score: float) -> bool:
        diff = new_score - old_score
        if torch.log(torch.rand(1)) < diff: return True
        return False

    def __acceptance_ratio(self) -> float: return self.__accepted / self.__steps

