import time
from datetime import datetime
from typing import Callable, Any
from src.utils.Logger import Logger


class Timer:
    """
    A utility class for timing the execution of methods.

    This class can be used as a decorator to time the execution of class methods
    and log the timing information to a logger.

    Attributes:
        logger (Logger): An instance of a logger that has a `log` method for logging messages.
    """

    def __init__(self, logger: Logger) -> None:
        """
        Initializes the Timer with a logger instance.

        Args:
            logger: An instance of a logger that has a `log` method for logging messages.
        """
        self.logger = logger

    def _decorator(self, func: Callable) -> Callable:
        """
        Decorates a function to log its execution time.

        This method wraps the provided function, measures the time it takes to execute,
        and logs the timing information using the provided logger.

        Args:
            func: The function to be decorated.

        Returns:
            Callable: The wrapped function with timing logic.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            A wrapper function to time the execution of the decorated function.

            This function records the start time before calling the decorated function,
            calculates the elapsed time after the function completes, and logs the
            timing information using the provided logger.

            Args:
                *args: Positional arguments to pass to the decorated function.
                **kwargs: Keyword arguments to pass to the decorated function.

            Returns:
                Any: The result returned by the decorated function.
            """
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            hours, remainder = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_statement = f"{func.__name__} executed in {int(hours):02}h {int(minutes):02}m {seconds:.2f}s"

            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{time_statement} at {current_datetime}"

            self.logger.log("-----------------------")
            self.logger.log(log_message)
            self.logger.log("-----------------------")

            return result

        return wrapper
