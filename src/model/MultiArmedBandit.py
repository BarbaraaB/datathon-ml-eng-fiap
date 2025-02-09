import numpy as np


class MultiArmedBandit:
    def __init__(self, n_arms):
        """
        Inicializa o MAB com um número fixo de braços (notícias).

        :param n_arms: Número de notícias (braços) disponíveis.
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Número de vezes que cada braço foi escolhido
        self.values = np.zeros(n_arms)  # Recompensa média estimada para cada braço

    def select_arm(self, exploration_rate=0.1):
        if np.random.rand() < exploration_rate:
            return np.random.choice(self.n_arms)
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            return np.random.choice(self.n_arms)
        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        Atualiza a recompensa média e o contador para o braço escolhido.

        :param chosen_arm: Índice do braço escolhido.
        :param reward: Recompensa observada (ex: número de cliques).
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
