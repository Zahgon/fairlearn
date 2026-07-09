

class ThresholdOperation:

    def __init__(self, operator, threshold):
        if operator not in [">", "<"]:
            raise ValueError("Unrecognized operator: " + operator)
        self._operator = operator
        self._threshold = threshold

    @property
    def threshold(self):
        pass

    @property
    def operator(self):
        pass

    def __call__(self, y_hat):
        """Evaluate the threshold rule `y_hat > t` or `y_hat < t`.

        Parameters
        ----------
        y_hat : array
            The input array.

        Returns
        -------
        y_hat : array
            The result of elementwise application of the threshold rule.
        """
        if self._operator == ">":
            return y_hat > self._threshold
        elif self._operator == "<":
            return y_hat < self._threshold
        else:
            raise ValueError("Unrecognized operator: " + self._operator)

    def __repr__(self):
        return "[{}{}]".format(self._operator, self._threshold)
