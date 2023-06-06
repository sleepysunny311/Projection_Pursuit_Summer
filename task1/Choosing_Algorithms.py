import numpy as np

class AlphaChoosingClass:
    def __init__(self, dictionary, ratio=0.5, seed=123, with_replace_flag=True):
        """
        Initialize the AlphaChoosingClass.

        Args:
            dictionary: A numpy array or matrix representing the data.
            ratio: The ratio of selected rows to total rows.
            seed: Seed value for the random number generator.
            with_replace_flag: Flag indicating whether to sample with replacement.
        """
        self.dictionary = dictionary
        self.ratio = ratio
        self.random_state = np.random.default_rng(seed)
        self.selected_rows = self.choose_dimensions(with_replace_flag)

    def choose_dimensions(self, with_replace_flag=False):
        """
        Choose a subset of rows from the data based on the specified ratio.

        Args:
            with_replace_flag: Flag indicating whether to sample with replacement.

        Returns:
            A numpy array containing the indices of the selected rows.
        """
        num_rows = self.dictionary.shape[0]
        num_selected_rows = int(self.ratio * num_rows)

        """
        Generate a random sample of row indices
        """
        selected_rows = self.random_state.choice(num_rows, size=num_selected_rows, replace=with_replace_flag)

        return selected_rows

    def get_selected_rows(self):
        """
        Get the indices of the selected rows.

        Returns:
            A numpy array containing the indices of the selected rows.
        """
        return self.selected_rows

    def set_ratio(self, new_ratio):
        """
        Set a new ratio for selecting rows and update the selected rows accordingly.

        Args:
            new_ratio: The new ratio of selected rows to total rows.
        """
        self.ratio = new_ratio
        self.selected_rows = self.choose_dimensions()

    def set_seed(self, new_seed):
        """
        Set a new seed value for the random number generator.

        Args:
            new_seed: The new seed value.
        """
        self.random_state = np.random.default_rng(new_seed)
        self.selected_rows = self.choose_dimensions()

    def set_with_replace_flag(self, flag):
        """
        Set the with_replace_flag.

        Args:
            flag: Flag indicating whether to sample with replacement.
        """
        self.selected_rows = self.choose_dimensions(with_replace_flag=flag)
        self.selected_rows = self.choose_dimensions()
        
    def report(self):
        """
        Print the currently selected rows and the current ratio.
        """
        print(f"Currently selected rows: {self.get_selected_rows()}")
        print(f"Current ratio: {self.ratio}")

    def __str__(self):
        """
        Return a string representation of the AlphaChoosingClass.

        Returns:
            str: String representation of the AlphaChoosingClass.
        """
        return f"Currently selected rows: {self.get_selected_rows()}\nCurrent ratio: {self.ratio}"
