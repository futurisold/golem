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

    def log_likelihood(self):
        '''Compute the log likelihood'''
        params = {}
        for prior in self.priors.values(): params |= {prior.estimand: prior.theta}

        return self.model(**params).log_prob(self.data).sum()

    def log_posterior(self):
        '''Compute the log posterior'''
        total = 0
        for prior in self.priors.values(): total += prior.dist.log_prob(prior.theta)

        return total + self.log_likelihood()

    def maximum_a_posteriori(self, lr: float = 1e-1, n_iter: int = 1000):
        '''Find the MAP estimate of the parameters'''
        optim = torch.optim.Adam([prior.theta for prior in self.priors.values()], lr=lr)
        for _ in range(n_iter):
            optim.zero_grad()
            loss = -self.log_posterior()
            loss.backward()
            optim.step()

        return {prior.estimand: prior.theta.item() for prior in self.priors.values()}


class Golem:
    def __init__(self,
                 assumptions: dict[str, str] | None = None) -> None:
        self.assumptions = assumptions

    def build_dag(self, show=False) -> nx.Graph:
        '''
        Builds a DAG of the model.
        '''
        assert self.assumptions is not None, "No generative assumptions provided"
        graph = nx.DiGraph()
        for fr, to in self.assumptions.items():
            graph.add_edge(fr, to)

        if show:
            nx.draw(graph, arrows=True, with_labels=True)
            plt.show()

        return graph

