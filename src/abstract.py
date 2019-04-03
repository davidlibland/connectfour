from abc import ABC, abstractmethod
from typing import Iterator, Optional, Generic, TypeVar, Type, List, Union

P = TypeVar("P")
A = TypeVar("A")

class AbsBatchGameState(ABC, Generic[A, P]):
    @abstractmethod
    def next_actions(self) -> Iterator[A]:
        raise NotImplementedError

    @abstractmethod
    def winners(self) -> List[Optional[P]]:
        raise NotImplementedError

    @abstractmethod
    def play_at(self, A) -> "AbsBatchGameState":
        raise NotImplementedError

    @property
    @abstractmethod
    def turn(self) -> P:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    @classmethod
    def start_state(cls) -> "AbsBatchGameState":
        raise NotImplementedError


GS = TypeVar("GS", bound=AbsBatchGameState)
class ABSGame(ABC, Generic[GS]):
    @property
    @abstractmethod
    def cur_state(self) -> GS:
        raise NotImplementedError

    @property
    @abstractmethod
    def reversed_states(self) -> Iterator[GS]:
        raise NotImplementedError

    @abstractmethod
    def add_state(self, state: GS):
        raise NotImplementedError

    @abstractmethod
    def map(self, f) -> "ABSGame":
        raise NotImplementedError

    @classmethod
    def factory(cls, G: Type[GS]):
        class FGame(ABSGame):
            def __init__(self, **kwargs):
                self._liat = None
                self._cur_state = G(**kwargs)

            @property
            def cur_state(self) -> G:
                return self._cur_state

            @property
            def reversed_states(self) -> Iterator[G]:
                game = self
                while game is not None:
                    yield game._cur_state
                    game = game._liat

            def add_state(self, state: G):
                game = FGame()
                game._cur_state = state
                game._liat = self
                return game

            def map(self, f):
                cur = f(self._cur_state)
                if self._liat is not None:
                    liat = self._liat.map(f)
                    return liat.add_state(cur)
                g = FGame()
                g._cur_state = cur
                return g

        return FGame
