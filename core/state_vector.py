import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


class WuXingStateVector:
    """
    Five Elements State Vector for a machine learning system

    Represents the state of a model in terms of the Wu Xing (Five Elements) theory:
    - Water (水): Data quality and flow
    - Wood (木): Architecture complexity and growth
    - Fire (火): Training intensity and transformation
    - Earth (土): Regularization and stability
    - Metal (金): Evaluation precision and refinement
    """

    def __init__(self,
                 water: float = 0.5,
                 wood: float = 0.5,
                 fire: float = 0.5,
                 earth: float = 0.5,
                 metal: float = 0.5):
        """
        Initialize a Five Elements State Vector

        All parameters should be between 0 and 1, representing the relative
        strength of each element.

        Args:
            water: Strength of Water element (data quality)
            wood: Strength of Wood element (architecture complexity)
            fire: Strength of Fire element (training intensity)
            earth: Strength of Earth element (regularization)
            metal: Strength of Metal element (evaluation precision)
        """
        self.state = np.array([
            water,  # Water
            wood,  # Wood
            fire,  # Fire
            earth,  # Earth
            metal  # Metal
        ])

        # Ensure values are in valid range
        self.state = np.clip(self.state, 0, 1)

    def __repr__(self) -> str:
        """String representation of the state vector"""
        elements = ['Water (Data)', 'Wood (Architecture)',
                    'Fire (Training)', 'Earth (Regularization)',
                    'Metal (Evaluation)']

        return "\n".join([f"{elements[i]}: {self.state[i]:.2f}" for i in range(5)])

    def plot(self, ax: Optional[plt.Axes] = None, title: Optional[str] = None,
             color: str = 'blue', alpha: float = 0.25) -> plt.Figure:
        """
        Visualize the Wu Xing state as a radar chart

        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            title: Title for the plot. If None, uses default title.
            color: Color for the plot lines and fill
            alpha: Alpha value for the fill

        Returns:
            Matplotlib figure containing the plot
        """
        elements = ['Water (Data)', 'Wood (Architecture)',
                    'Fire (Training)', 'Earth (Regularization)',
                    'Metal (Evaluation)']

        angles = np.linspace(0, 2 * np.pi, len(elements), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        values = self.state.tolist()
        values += values[:1]  # Close the loop

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        else:
            fig = ax.figure

        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=alpha, color=color)
        ax.set_thetagrids(np.degrees(angles[:-1]), elements)
        ax.set_ylim(0, 1)
        ax.grid(True)

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Wu Xing State of Machine Learning System")

        return fig

    def balance_score(self) -> float:
        """
        Calculate overall balance score of the state

        Returns:
            A value between 0 and 1, where 1 is perfectly balanced
        """
        # Balance is highest when all elements are equal
        ideal = np.ones(5) * 0.6  # Optimal level for balance
        deviation = np.mean(np.abs(self.state - ideal))
        return 1.0 - deviation

    def generation_cycle_effect(self, time_step: float = 0.1) -> 'WuXingStateVector':
        """
        Apply the generation cycle effect (相生) to the state vector

        In Wu Xing theory, each element generates or nourishes the next:
        Water → Wood → Fire → Earth → Metal → Water

        Args:
            time_step: Size of the time step for the transformation

        Returns:
            New WuXingStateVector after generation cycle effects
        """
        # Create generation cycle matrix
        G = np.zeros((5, 5))

        # Set generation relationships:
        # Water (0) generates Wood (1)
        G[0, 1] = 1
        # Wood (1) generates Fire (2)
        G[1, 2] = 1
        # Fire (2) generates Earth (3)
        G[2, 3] = 1
        # Earth (3) generates Metal (4)
        G[3, 4] = 1
        # Metal (4) generates Water (0)
        G[4, 0] = 1

        # Calculate the generation effect
        generation_effect = np.zeros(5)
        for i in range(5):
            for j in range(5):
                if G[i, j] > 0:
                    # Element i generates element j
                    generation_effect[j] += self.state[i] * G[i, j] * 0.1

        # Update state with generation effects
        new_state = self.state + generation_effect * time_step

        # Normalize values to [0, 1]
        new_state = np.clip(new_state, 0, 1)

        # Create new state vector with updated values
        return WuXingStateVector(*new_state)

    def conquest_cycle_effect(self, time_step: float = 0.1) -> 'WuXingStateVector':
        """
        Apply the conquest cycle effect (相克) to the state vector

        In Wu Xing theory, each element constrains or controls another:
        Water → Fire → Metal → Wood → Earth → Water

        Args:
            time_step: Size of the time step for the transformation

        Returns:
            New WuXingStateVector after conquest cycle effects
        """
        # Create conquest cycle matrix
        C = np.zeros((5, 5))

        # Set conquest relationships:
        # Water (0) conquers Fire (2)
        C[0, 2] = 1
        # Fire (2) conquers Metal (4)
        C[2, 4] = 1
        # Metal (4) conquers Wood (1)
        C[4, 1] = 1
        # Wood (1) conquers Earth (3)
        C[1, 3] = 1
        # Earth (3) conquers Water (0)
        C[3, 0] = 1

        # Calculate the conquest effect
        conquest_effect = np.zeros(5)
        for i in range(5):
            for j in range(5):
                if C[i, j] > 0:
                    # Element i conquers element j
                    conquest_effect[j] -= self.state[i] * C[i, j] * 0.1

        # Update state with conquest effects
        new_state = self.state + conquest_effect * time_step

        # Normalize values to [0, 1]
        new_state = np.clip(new_state, 0, 1)

        # Create new state vector with updated values
        return WuXingStateVector(*new_state)

    def full_cycle_transformation(self, weights: dict, time_step: float = 0.1) -> 'WuXingStateVector':
        """
        Apply the complete Wu Xing transformation with weighted cycles

        Args:
            weights: Dictionary of weights for each cycle type
                - 'generation': Weight for generation cycle
                - 'conquest': Weight for conquest cycle
                - 'balance': Weight for tendency toward equilibrium
            time_step: Size of the time step for the transformation

        Returns:
            New WuXingStateVector after full transformation
        """
        # Copy current state
        new_state = self.state.copy()

        # Apply generation cycle
        if weights.get('generation', 0) > 0:
            gen_state = self.generation_cycle_effect(time_step).state
            new_state = new_state + weights['generation'] * (gen_state - self.state)

        # Apply conquest cycle
        if weights.get('conquest', 0) > 0:
            conq_state = self.conquest_cycle_effect(time_step).state
            new_state = new_state + weights['conquest'] * (conq_state - self.state)

        # Apply balance cycle (tendency toward equilibrium)
        if weights.get('balance', 0) > 0:
            balance_effect = np.zeros(5)
            mean_state = np.mean(self.state)
            for i in range(5):
                balance_effect[i] = (mean_state - self.state[i]) * weights['balance']
            new_state += balance_effect * time_step

        # Normalize values to [0, 1]
        new_state = np.clip(new_state, 0, 1)

        # Create new state vector with updated values
        return WuXingStateVector(*new_state)

    def element_names(self, chinese: bool = False) -> List[str]:
        """
        Get the names of the five elements

        Args:
            chinese: If True, include Chinese characters in the names

        Returns:
            List of element names
        """
        if chinese:
            return ['Water (水, Data)', 'Wood (木, Architecture)',
                    'Fire (火, Training)', 'Earth (土, Regularization)',
                    'Metal (金, Evaluation)']
        else:
            return ['Water (Data)', 'Wood (Architecture)',
                    'Fire (Training)', 'Earth (Regularization)',
                    'Metal (Evaluation)']