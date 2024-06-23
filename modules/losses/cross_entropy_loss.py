import numpy as np
from numpy import typing as npt


class CrossEntropyLoss:
    """Cross-Entropy loss with Softmax."""

    def __init__(self) -> None:
        """Initialize a new instance of CrossEntropyLoss."""

    def __call__(
        self,
        targets: npt.NDArray[np.floating],
        logits: npt.NDArray[np.floating],
    ) -> float:
        """Calculate cross-entropy loss.

        For a one-hot encoded targets t and model output y:
        E = - (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ln(y_k (x_i)),

        where:
        - N is the number of data points,
        - K is the number of classes,
        - t_{ik} is the value from OHE target matrix for data point i and
            class k,
        - y_k (x_i) is model output after softmax for data point i and class k.

        Numerically stable formula:
        E = (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1)
            t_ik * (ln(Σ(l=0 to K-1) e^((z_il - c_i)) - (z_ik - c_i)),

        where:
        - N is the number of data points,
        - K is the number of classes,
        - t_{ik} is the value from OHE target matrix for data point i and
            class k,
        - z_{il} is the model output before softmax for data point i and
            class l,
        - z is the model output before softmax (logits),
        - c_i is maximum value for each data point i in vector z_i.

        Parameters:
            targets (npt.NDArray[np.floating]): The one-hot encoded target
                data.
            logits (npt.NDArray[np.floating]): The model output before softmax.

        Returns:
            float: The value of the loss function.
        """
        max_logit = np.max(logits, axis=1, keepdims=True)
        logits -= max_logit
        exp_logits = np.exp(logits)
        sum_exp_logits = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        return np.mean(targets * (sum_exp_logits - logits))

    def backward(self, targets: np.ndarray, logits: np.ndarray):
        """Backward pass for Cross-Entropy Loss.

        For mini-batch, backward pass can be defined as follows:
            ∇_{Z^L} E = 1 / N (y - t)
            y = Softmax(Z^L)

        where:
            - Z^L - the model output before softmax
            - t (N x K matrix): One-Hot encoded targets representation

        Args:
            targets (np.ndarray): The one-hot encoded target data.
            logits (np.ndarray): The model output before softmax.

        Returns:
            ∇_{Z^L} E: matrix of shape (batch_size, K)
        """
        # TODO: Implement Cross-Entropy Loss backward propagation
        raise NotImplementedError
