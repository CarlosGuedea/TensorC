"""NumPy implementation of TensorC's vector/discrete-action DQN API."""

from __future__ import annotations

import operator
from pathlib import Path

import numpy as np


def _positive_int(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s debe ser un entero" % name)
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError("%s debe ser un entero" % name) from exc
    if value < minimum:
        raise ValueError("%s debe ser >= %d" % (name, minimum))
    return value


def _finite_float(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s debe ser numérico" % name) from exc
    if not np.isfinite(value):
        raise ValueError("%s debe ser finito" % name)
    return value


def _flag(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("%s debe ser booleano" % name)
    return bool(value)


class DQNAgent:
    """DQN NumPy para observaciones vectoriales y acciones discretas.

    ``train_step`` devuelve la pérdida MSE, o ``None`` durante el warm-up del
    replay. ``truncated`` termina el episodio, pero sigue permitiendo bootstrap;
    ``terminated`` lo detiene.

    ``save`` guarda el modelo para inferencia. No guarda replay ni estado Adam,
    por lo que ``load`` no reanuda exactamente una sesión de entrenamiento.
    """

    _FORMAT_VERSION = 1

    def __init__(self, observation_size, action_count, *, hidden_size=64,
                 replay_capacity=10000, batch_size=64,
                 target_update_interval=250, learning_rate=0.001,
                 discount_factor=0.99, seed=1):
        self.observation_size = _positive_int("observation_size", observation_size)
        self.action_count = _positive_int("action_count", action_count, minimum=2)
        self.hidden_size = _positive_int("hidden_size", hidden_size)
        self.replay_capacity = _positive_int("replay_capacity", replay_capacity)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.target_update_interval = _positive_int(
            "target_update_interval", target_update_interval
        )
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size no puede superar replay_capacity")
        self.learning_rate = _finite_float("learning_rate", learning_rate)
        self.discount_factor = _finite_float("discount_factor", discount_factor)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate debe ser positivo")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor debe estar en [0, 1]")
        if seed is None:
            self.seed = None
        else:
            self.seed = _positive_int("seed", seed, minimum=0)

        self._rng = np.random.default_rng(self.seed)
        self._online = [
            self._rng.normal(0.0, np.sqrt(2.0 / self.observation_size),
                            (self.observation_size, self.hidden_size)).astype(np.float32),
            np.zeros(self.hidden_size, dtype=np.float32),
            self._rng.normal(0.0, np.sqrt(2.0 / self.hidden_size),
                            (self.hidden_size, self.action_count)).astype(np.float32),
            np.zeros(self.action_count, dtype=np.float32),
        ]
        self._target = [value.copy() for value in self._online]
        self._adam_m = [np.zeros_like(value) for value in self._online]
        self._adam_v = [np.zeros_like(value) for value in self._online]
        self._adam_step = 0

        self._states = np.empty(
            (self.replay_capacity, self.observation_size), dtype=np.float32
        )
        self._next_states = np.empty_like(self._states)
        self._actions = np.empty(self.replay_capacity, dtype=np.int64)
        self._rewards = np.empty(self.replay_capacity, dtype=np.float32)
        self._terminated = np.empty(self.replay_capacity, dtype=np.bool_)
        self._replay_size = 0
        self._replay_next = 0
        self._train_steps = 0

    @property
    def replay_size(self):
        return self._replay_size

    def _observation(self, value, name):
        try:
            result = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("%s debe contener valores numéricos" % name) from exc
        if result.shape != (self.observation_size,):
            raise ValueError(
                "%s debe tener forma (%d,)" % (name, self.observation_size)
            )
        if not np.isfinite(result).all():
            raise ValueError("%s debe contener solo valores finitos" % name)
        return np.ascontiguousarray(result)

    @staticmethod
    def _forward(states, parameters):
        w1, b1, w2, b2 = parameters
        hidden_pre = states @ w1 + b1
        hidden = np.maximum(hidden_pre, 0.0)
        return hidden @ w2 + b2, hidden_pre, hidden

    def select_action(self, observation, epsilon=0.0):
        """Elige una acción epsilon-greedy y devuelve su índice entero."""
        state = self._observation(observation, "observation")
        epsilon = _finite_float("epsilon", epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon debe estar en [0, 1]")
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.action_count))
        q_values, _, _ = self._forward(state[None, :], self._online)
        if not np.isfinite(q_values).all():
            raise FloatingPointError("la red produjo valores Q no finitos")
        return int(np.argmax(q_values[0]))

    def remember(self, observation, action, reward, next_observation,
                 terminated, truncated=False):
        """Guarda una transición; solo ``terminated`` impide bootstrap."""
        state = self._observation(observation, "observation")
        next_state = self._observation(next_observation, "next_observation")
        if isinstance(action, (bool, np.bool_)):
            raise TypeError("action debe ser un entero")
        try:
            action = operator.index(action)
        except TypeError as exc:
            raise TypeError("action debe ser un entero") from exc
        if not 0 <= action < self.action_count:
            raise ValueError("action fuera del espacio discreto")
        reward = _finite_float("reward", reward)
        terminated = _flag("terminated", terminated)
        _flag("truncated", truncated)

        index = self._replay_next
        self._states[index] = state
        self._next_states[index] = next_state
        self._actions[index] = action
        self._rewards[index] = reward
        self._terminated[index] = terminated
        self._replay_next = (index + 1) % self.replay_capacity
        self._replay_size = min(self._replay_size + 1, self.replay_capacity)

    def train_step(self):
        """Actualiza con un minibatch; devuelve ``None`` durante warm-up."""
        if self._replay_size < self.batch_size:
            return None
        indices = self._rng.integers(self._replay_size, size=self.batch_size)
        states = self._states[indices]
        next_states = self._next_states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        terminated = self._terminated[indices].astype(np.float32)

        current_q, hidden_pre, hidden = self._forward(states, self._online)
        next_q, _, _ = self._forward(next_states, self._target)
        targets = rewards + (1.0 - terminated) * self.discount_factor * np.max(
            next_q, axis=1
        )
        rows = np.arange(self.batch_size)
        errors = current_q[rows, actions] - targets
        loss = float(np.mean(errors * errors))
        if (not np.isfinite(current_q).all() or not np.isfinite(next_q).all() or
                not np.isfinite(targets).all() or not np.isfinite(loss)):
            raise FloatingPointError("DQN produjo Q, targets o pérdida no finitos")

        grad_q = np.zeros_like(current_q)
        grad_q[rows, actions] = (2.0 / self.batch_size) * errors
        w1, _, w2, _ = self._online
        grad_w2 = hidden.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)
        grad_hidden = grad_q @ w2.T
        grad_hidden_pre = grad_hidden * (hidden_pre > 0.0)
        grad_w1 = states.T @ grad_hidden_pre
        grad_b1 = grad_hidden_pre.sum(axis=0)
        gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
        if not all(np.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("DQN produjo gradientes no finitos")

        self._adam_step += 1
        beta1, beta2, adam_epsilon = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1 ** self._adam_step
        correction2 = 1.0 - beta2 ** self._adam_step
        candidates = []
        moments = []
        variances = []
        for parameter, gradient, moment, variance in zip(
            self._online, gradients, self._adam_m, self._adam_v
        ):
            next_moment = beta1 * moment + (1.0 - beta1) * gradient
            next_variance = beta2 * variance + (1.0 - beta2) * gradient * gradient
            updated = parameter - self.learning_rate * (
                next_moment / correction1
            ) / (np.sqrt(next_variance / correction2) + adam_epsilon)
            if (not np.isfinite(next_moment).all() or
                    not np.isfinite(next_variance).all() or
                    not np.isfinite(updated).all()):
                raise FloatingPointError("actualización Adam no finita")
            moments.append(next_moment)
            variances.append(next_variance)
            candidates.append(updated.astype(np.float32, copy=False))

        for parameter, updated, moment, variance, new_moment, new_variance in zip(
            self._online, candidates, self._adam_m, self._adam_v,
            moments, variances
        ):
            parameter[...] = updated
            moment[...] = new_moment
            variance[...] = new_variance

        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self.sync_target()
        return loss

    def sync_target(self):
        """Copia los parámetros online a la red objetivo."""
        for target, online in zip(self._target, self._online):
            target[...] = online

    def save(self, path):
        """Guarda un checkpoint de inferencia NumPy sin pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez(
                output,
                format_version=np.asarray(self._FORMAT_VERSION, dtype=np.int64),
                observation_size=np.asarray(self.observation_size, dtype=np.int64),
                action_count=np.asarray(self.action_count, dtype=np.int64),
                hidden_size=np.asarray(self.hidden_size, dtype=np.int64),
                replay_capacity=np.asarray(self.replay_capacity, dtype=np.int64),
                batch_size=np.asarray(self.batch_size, dtype=np.int64),
                target_update_interval=np.asarray(
                    self.target_update_interval, dtype=np.int64
                ),
                learning_rate=np.asarray(self.learning_rate, dtype=np.float64),
                discount_factor=np.asarray(self.discount_factor, dtype=np.float64),
                seed=np.asarray(-1 if self.seed is None else self.seed, dtype=np.int64),
                w1=self._online[0], b1=self._online[1],
                w2=self._online[2], b2=self._online[3],
            )

    @classmethod
    def load(cls, path):
        """Carga pesos y configuración para inferencia; replay queda vacío."""
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["format_version"]) != cls._FORMAT_VERSION:
                raise ValueError("versión de checkpoint DQN no compatible")
            seed_value = int(checkpoint["seed"])
            agent = cls(
                int(checkpoint["observation_size"]),
                int(checkpoint["action_count"]),
                hidden_size=int(checkpoint["hidden_size"]),
                replay_capacity=int(checkpoint["replay_capacity"]),
                batch_size=int(checkpoint["batch_size"]),
                target_update_interval=int(checkpoint["target_update_interval"]),
                learning_rate=float(checkpoint["learning_rate"]),
                discount_factor=float(checkpoint["discount_factor"]),
                seed=None if seed_value < 0 else seed_value,
            )
            names = ("w1", "b1", "w2", "b2")
            for parameter, name in zip(agent._online, names):
                value = np.asarray(checkpoint[name], dtype=np.float32)
                if value.shape != parameter.shape or not np.isfinite(value).all():
                    raise ValueError("forma o valores inválidos en checkpoint DQN")
                parameter[...] = value
        agent.sync_target()
        return agent
