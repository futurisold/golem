import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.distributions import Distribution


class Prior:
    def __init__(self, dist: Distribution, **params: float):
        self.dist = dist(**params)
        assert self.dist.has_rsample, "Distribution must have the reparameterization trick"
        super().__setattr__('theta', self.__sample_learable_param())

    def __sample_learable_param(self): return self.dist.log_prob(self.dist.rsample())


class Model:
    def __init__(self,
                 data: torch.Tensor,
                 model: Distribution,
                 **priors: Prior):
        self.data   = data
        self.model  = model
        self.priors = priors


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

