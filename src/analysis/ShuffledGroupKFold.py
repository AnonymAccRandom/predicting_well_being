from typing import Optional, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import shuffle


class ShuffledGroupKFold(BaseCrossValidator):
    """
    Custom GroupKFold class with shuffling and random state support.

    This class extends the functionality of group-based cross-validation by adding the ability to shuffle groups
    before splitting. The groups are shuffled based on the provided random state to ensure reproducibility.

    Attributes:
        n_splits (int): Number of splits/folds for cross-validation.
        shuffle (bool): Whether to shuffle the groups before splitting.
        random_state (Optional[int]): Seed for the random number generator for shuffling.
    """

    def __init__(
        self, n_splits: int, random_state: Optional[int], shuffle: bool = True
    ) -> None:
        """
        Initialize the ShuffledGroupKFold instance.

        Args:
            n_splits: Number of folds for cross-validation.
            random_state: Seed for the random number generator used in shuffling.
            shuffle: Whether to shuffle the groups before splitting. Defaults to True.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Not used, present for compatibility.
            y: Target vector of shape (n_samples,). Not used, present for compatibility.
            groups: Group labels for the samples of shape (n_samples,). Not used, present for compatibility.

        Returns:
            int: The number of splits (folds).
        """
        return self.n_splits

    def split(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets based on group labels.

        This method implements group-based cross-validation where samples in the same group
        are kept together in the training or test sets. If shuffling is enabled, group labels
        are shuffled before splitting into folds.

        Args:
            X (Optional[np.ndarray]): Feature matrix of shape (n_samples, n_features). Not used, present for compatibility.
            y (Optional[np.ndarray]): Target vector of shape (n_samples,). Not used, present for compatibility.
            groups (Optional[np.ndarray]): Group labels for the samples of shape (n_samples,). Determines the grouping
                for the splits. Each unique value in `groups` represents a distinct group.

        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: An iterator of tuples, where each tuple contains:
                - `train_indices` (np.ndarray): Indices for the training set.
                - `test_indices` (np.ndarray): Indices for the test set.
        """
        groups = pd.Series(groups)
        ix = np.arange(len(groups))
        unique = np.unique(groups)

        np.random.RandomState(self.random_state).shuffle(unique)
        result = []

        for split in np.array_split(unique, self.n_splits):
            mask = groups.isin(split)
            train, test = ix[~mask], ix[mask]
            result.append((train, test))

        return result
