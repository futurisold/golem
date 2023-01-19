from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.distributions import Distribution


class Prior:
    def __init__(self, estimand: str, dist: Distribution, **params: float):
        self.dist = dist(**params)
        self.estimand = estimand
        # Initialize the state of the prior
        self.theta = torch.ones(1, requires_grad=True)


class Model:
    def __init__(self,
                 data: torch.Tensor,
                 model: Distribution,
                 **priors: Prior):
        self.data   = data
        self.model  = model
        self.priors = priors

    def log_prior(self):
        return sum(prior.dist.log_prob(prior.theta) for prior in self.priors.values())

    def log_likelihood(self):
        params = self.iter_params()
        return sum(self.model(**params).log_prob(self.data))

    def log_posterior(self):
        return self.log_prior() + self.log_likelihood()

    def iter_params(self):
        params = {}
        for prior in self.priors.values(): params |= {prior.estimand: prior.theta}
        return params

    def update(self, estimand: str, theta: torch.Tensor):
        for prior in self.priors.values():
            # skip if theta is not in the support of the prior
            if prior.estimand == estimand and prior.dist.support.check(theta): prior.theta = theta


class Golem:
    def __init__(self,
                 models: list[Model],
                 assumptions: dict[str, str] | None = None) -> None:
        self.models = models
        self.assumptions = assumptions

    def build_dag(self, show=False) -> nx.Graph:
        assert self.assumptions is not None, "No generative assumptions provided"
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


class MetropolisHastings:
    def __init__(self,
                 models: list[Model]) -> None:

        self.models  = models
        # keep it simple and naive - use a normal distribution
        self.sampler = torch.distributions.Normal(0, 1)
        self.__steps = 0
        # keep track of the accepted steps to tune the scale of the sampler every 100 steps
        self.__accepted = 0
        self.__tune_interval = 100
        self.__tune_countdown = self.__tune_interval

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
                self.sampler.loc = param
                proposal = self.sampler.sample(param.shape)
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
        if ratio < 0.001:  self.sampler.scale *= 0.1 # reduce by 90%
        elif ratio < 0.05: self.sampler.scale *= 0.5 # reduce by 50%
        elif ratio < 0.20: self.sampler.scale *= 0.9 # reduce by 10%
        elif ratio > 0.95: self.sampler.scale *= 10  # increase by 1000%
        elif ratio > 0.75: self.sampler.scale *= 2   # increase by 100%
        elif ratio > 0.50: self.sampler.scale *= 1.1 # increase by 10%

    def __accept_proposal(self, new_score: float, old_score: float) -> bool:
        diff = new_score - old_score
        if torch.log(torch.rand(1)) < diff: return True
        return False

    def __acceptance_ratio(self) -> float: return self.__accepted / self.__steps

