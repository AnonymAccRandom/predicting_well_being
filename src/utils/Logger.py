import os
from datetime import datetime


class Logger:
    """
    A singleton logger class to handle logging messages to separate log files.

    This logger:
    - Ensures thread-safe logging using a combination of locks and unique keys based on `rank` and `rep`.
    - Creates a unique log file for each logger instance identified by the combination of `rank` and `rep`.
    - Writes log entries to the specified log file with timestamps.

    Attributes:
        log_file (str): The path to the log file for this logger instance.
    """

    _instances: dict[tuple[int, str], "Logger"] = {}
    _instance_lock: dict[tuple[int, str], bool] = {}

    def __new__(
        cls,
        log_dir: str = "tests",
        log_file: str = "log",
        rank: int = 0,
        rep: int = 0,
    ) -> "Logger":
        """
        Creates or retrieves a singleton instance of Logger based on the combination of `rank` and `rep`.

        Args:
            log_dir: The directory where log files will be stored. Defaults to 'tests'.
            log_file: The base name of the log file. Defaults to 'log'.
            rank: The rank identifier for this logger instance (e.g., process rank in distributed systems). Defaults to 0.
            rep: The repetition identifier for this logger instance. Defaults to "all".

        Returns:
            Logger: The singleton logger instance for the given `rank` and `rep`.
        """
        key = (rank, rep)

        # Ensure thread-safe access to _instances
        if key not in cls._instances:
            # Lock per key to prevent race conditions between processes
            cls._instance_lock[key] = cls._instance_lock.get(key, False)

            if not cls._instance_lock[key]:
                cls._instance_lock[key] = True
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                    log_file_with_rank_rep = (
                        f"{log_file}_{current_time}_rank{rank}_rep{rep}.log"
                    )
                    log_file_path = os.path.join(log_dir, log_file_with_rank_rep)

                    instance = super(Logger, cls).__new__(cls)
                    instance.log_file = log_file_path
                    instance._initialize_log()

                    cls._instances[key] = instance

                finally:
                    cls._instance_lock[key] = False

        return cls._instances[key]

    def _initialize_log(self) -> None:
        """
        Initializes the log file by writing a header indicating the logger's initialization time.
        """
        with open(self.log_file, "a") as file:
            file.write("-----------------------------------------------\n")
            file.write(
                f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            )
            file.write("-----------------------------------------------\n")

    def log(self, message: str) -> None:
        """
        Writes a log message to the log file with a timestamp.

        Args:
            message: The log message to write.
        """
        with open(self.log_file, "a") as file:
            file.write(f"### - {message}\n")
