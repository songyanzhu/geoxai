import os


def is_colab_hosted(*, verbose: bool = False) -> bool:
    """Detect whether the runtime is hosted by Google Colab.

    Args:
        verbose: Whether to print the runtime type. Defaults to False.

    Returns:
        True if Colab hosted runtime indicators are present, otherwise False.
    """
    is_hosted = (
        "COLAB_RELEASE_TAG" in os.environ
        or "COLAB_BACKEND_VERSION" in os.environ
    )

    if verbose:
        print("Colab hosted runtime" if is_hosted else "Local runtime")

    return is_hosted