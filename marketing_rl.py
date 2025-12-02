"""
Demo de aprendizado por reforço aplicado ao dataset fornecido.

O roteiro segue o briefing do arquivo "Aprendizado por Reforço.docx":
    * Criar um ambiente de marketing a partir do histórico do dataset
    * Definir recompensas que valorizem conversões e penalizem custos
    * Treinar um agente de RL tabular (Q-Learning) para escolher ações
    * Comparar a política treinada com um baseline aleatório
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def load_and_prepare_dataset(path: str) -> pd.DataFrame:
    """Carrega o CSV e deriva atributos de estado/recompensa para o simulador."""
    df = pd.read_csv(path).copy()

    # Normaliza a coluna de conversão (já é 0/1, mas garantimos inteiro)
    df["Conversion"] = df["Conversion"].astype(int)

    # Discretiza descritores principais dos clientes que formam o estado
    age_bins = sorted(
        set([df["Age"].min() - 1, 25, 35, 45, 55, 65, df["Age"].max() + 1])
    )
    df["AgeBucket"] = pd.cut(
        df["Age"], bins=age_bins, labels=False, include_lowest=True
    ).astype(int)

    income_bins = min(5, df["Income"].nunique())
    df["IncomeBucket"] = pd.qcut(
        df["Income"], q=income_bins, labels=False, duplicates="drop"
    ).astype(int)

    gender_map = {"Female": 0, "Male": 1}
    df["GenderCode"] = df["Gender"].map(gender_map).fillna(2).astype(int)

    state_cols = ["AgeBucket", "IncomeBucket", "GenderCode"]
    df["StateKey"] = list(zip(*(df[col] for col in state_cols)))

    # A recompensa combina proxy de receita (conversão) com penalidade de custo.
    # Os coeficientes garantem que uma conversão (~120 unidades) supere
    # o custo médio de AdSpend (~12 unidades após a divisão por 400).
    df["Reward"] = (
        df["Conversion"] * 120.0
        + df["ConversionRate"] * 30.0
        + df["ClickThroughRate"] * 10.0
        - df["AdSpend"] / 400.0
    )

    return df


@dataclass
class StepInfo:
    customer_id: int
    action: str
    conversion: int
    ad_spend: float


class MarketingEnv:
    """Ambiente estilo contextual bandit baseado no histórico de marketing."""

    def __init__(self, df: pd.DataFrame, max_steps: int = 25, seed: int = 42) -> None:
        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.base_seed = seed
        self.rng = random.Random(seed)

        self.actions: List[str] = sorted(self.df["CampaignChannel"].unique())
        self.state_keys: List[Tuple[int, int, int]] = sorted(self.df["StateKey"].unique())
        self.state_to_idx: Dict[Tuple[int, int, int], int] = {
            key: idx for idx, key in enumerate(self.state_keys)
        }
        self.action_to_idx: Dict[str, int] = {
            act: idx for idx, act in enumerate(self.actions)
        }

        self.state_lookup: Dict[int, List[int]] = defaultdict(list)
        self.state_action_lookup: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for idx, row in self.df.iterrows():
            state_idx = self.state_to_idx[row["StateKey"]]
            action_idx = self.action_to_idx[row["CampaignChannel"]]
            self.state_lookup[state_idx].append(idx)
            self.state_action_lookup[(state_idx, action_idx)].append(idx)

        self.all_indices = list(range(len(self.df)))
        self.current_state = 0
        self.steps = 0

    @property
    def num_states(self) -> int:
        return len(self.state_keys)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def reseed(self, seed: int | None = None) -> None:
        """Reinicializa o gerador aleatório para reprodutibilidade."""
        self.rng.seed(self.base_seed if seed is None else seed)

    def reset(self) -> int:
        self.steps = 0
        self.current_state = self.rng.randrange(self.num_states)
        return self.current_state

    def _sample_row(self, state_idx: int, action_idx: int) -> pd.Series:
        key = (state_idx, action_idx)
        candidates = self.state_action_lookup.get(key)

        if candidates:
            picked = self.rng.choice(candidates)
        elif self.state_lookup[state_idx]:
            picked = self.rng.choice(self.state_lookup[state_idx])
        else:
            picked = self.rng.choice(self.all_indices)

        return self.df.iloc[picked]

    def step(self, action_idx: int) -> Tuple[int, float, bool, StepInfo]:
        row = self._sample_row(self.current_state, action_idx)
        reward = float(row["Reward"])
        info = StepInfo(
            customer_id=int(row["CustomerID"]),
            action=row["CampaignChannel"],
            conversion=int(row["Conversion"]),
            ad_spend=float(row["AdSpend"]),
        )

        self.steps += 1
        done = self.steps >= self.max_steps
        self.current_state = self.rng.randrange(self.num_states)
        return self.current_state, reward, done, info


def train_q_learning(
    env: MarketingEnv,
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, List[float]]:
    """Loop padrão de Q-Learning tabular."""
    random.seed(42)
    q_table = np.zeros((env.num_states, env.num_actions), dtype=float)
    episode_rewards: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0.0

        while not done:
            if random.random() < epsilon:
                action = random.randrange(env.num_actions)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done, _ = env.step(action)
            best_future = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * best_future
            )
            state = next_state
            cumulative_reward += reward

        episode_rewards.append(cumulative_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_rewards


def evaluate_policy(
    env: MarketingEnv,
    policy: str,
    q_table: np.ndarray | None,
    episodes: int = 200,
    seed_offset: int = 0,
) -> Dict[str, float]:
    """Simula o ambiente para uma política e coleta métricas."""
    env.reseed(seed=env.base_seed + seed_offset)
    rewards: List[float] = []
    conversion_rates: List[float] = []
    action_frequencies: List[np.ndarray] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        conversions = 0
        steps = 0
        counts = np.zeros(env.num_actions, dtype=float)

        while not done:
            if policy == "random":
                action = random.randrange(env.num_actions)
            elif policy == "greedy":
                assert q_table is not None
                action = int(np.argmax(q_table[state]))
            else:  # política epsilon-greedy com leve exploração
                assert q_table is not None
                if random.random() < 0.1:
                    action = random.randrange(env.num_actions)
                else:
                    action = int(np.argmax(q_table[state]))

            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            conversions += info.conversion
            steps += 1
            counts[action] += 1

        rewards.append(total_reward)
        conversion_rates.append(conversions / max(steps, 1))
        action_frequencies.append(counts / max(steps, 1))

    avg_action_mix = np.mean(np.vstack(action_frequencies), axis=0)
    mix = {env.actions[i]: round(float(avg_action_mix[i]), 3) for i in range(env.num_actions)}

    return {
        "policy": policy,
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "avg_conversion_rate": float(np.mean(conversion_rates)),
        "action_mix": mix,
    }


def run_pipeline(
    data_path: str,
    episodes: int,
    max_steps: int,
) -> Dict[str, Dict[str, float]]:
    df = load_and_prepare_dataset(data_path)
    env = MarketingEnv(df, max_steps=max_steps, seed=42)
    q_table, history = train_q_learning(env, episodes=episodes)

    # Mantém um resumo leve do treinamento (média móvel das recompensas)
    history_series = pd.Series(history)
    moving = history_series.rolling(window=20).mean().iloc[-1]

    greedy_eval = evaluate_policy(env, policy="greedy", q_table=q_table, seed_offset=7)
    random_eval = evaluate_policy(env, policy="random", q_table=None, seed_offset=21)

    return {
        "dataset_rows": len(df),
        "num_states": env.num_states,
        "num_actions": env.num_actions,
        "training_reward_last20": float(moving),
        "greedy_eval": greedy_eval,
        "random_eval": random_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um agente de RL para marketing.")
    parser.add_argument(
        "--data",
        default="digital_marketing_campaign_dataset.csv",
        help="Caminho para o CSV de marketing.",
    )
    parser.add_argument(
        "--episodes", type=int, default=600, help="Número de episódios de Q-Learning."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Número de decisões de marketing por episódio simulado.",
    )
    args = parser.parse_args()

    results = run_pipeline(args.data, args.episodes, args.max_steps)

    print("Estatisticas do dataset:")
    print(
        f"  linhas={results['dataset_rows']} | estados={results['num_states']} |"
        f" acoes={results['num_actions']}"
    )
    print(f"Media movel das recompensas (ultimos 20 episodios): {results['training_reward_last20']:.2f}")
    print("\nAvaliacao de politicas:")
    nomes_politicas = {"random": "aleatoria", "greedy": "gananciosa"}
    for label in ("random_eval", "greedy_eval"):
        metrics = results[label]
        nome = nomes_politicas.get(metrics["policy"], metrics["policy"])
        print(f"- politica {nome}:")
        print(f"    recompensa media: {metrics['avg_reward']:.2f} +/- {metrics['reward_std']:.2f}")
        print(f"    taxa media de conversao: {metrics['avg_conversion_rate']:.3f}")
        print(f"    distribuicao de acoes: {metrics['action_mix']}")


if __name__ == "__main__":
    main()
