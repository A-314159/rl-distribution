from dataclasses import dataclass

# ---------------------------------------------
# At target, universe class should include anything related to the pseudo-Markovian state, such as:
# a) time, multi-dimension variable of a stochastic process that represent the state
# b) in finance: the description of the portfolio to be hedged (and the frequency of actions)
# c) a model of evolution of the universe
# ---------------------------------------------

@dataclass
class UniverseBS:
    sigma: float
    T: float
    P: int
    K: float = 1.0
    @property
    def h(self): return self.T / self.P
