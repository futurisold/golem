import matplotlib.pyplot as plt
import networkx as nx
import torch
from beartype import beartype
from torch.distributions import Distribution


@beartype
class Golem:
    def __init__(self,
                 statistical_model: Distribution,
                 generative_assumptions: dict[str, str] | None = None) -> None:
        self.generative_assumptions = generative_assumptions
        self.statistical_model = statistical_model

    def build_dag(self, show=False) -> nx.Graph:
        assert self.generative_assumptions is not None, "No generative assumptions provided"
        graph = nx.DiGraph()
        for fr, to in self.generative_assumptions.items():
            graph.add_edge(fr, to)

        if show:
            nx.draw(graph, arrows=True, with_labels=True)
            plt.show()

        return graph

    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        return self.statistical_model.sample(size)

