# Standard library imports
import os
import shutil
import asyncio
import logging
from typing import Annotated

# Related third-party imports
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Logger:
    """
    Logger class to simplify logging setup and usage.

    This class provides a reusable logging utility that supports customizable
    log levels, message formatting, and optional console printing.

    Parameters
    ----------
    name : str, optional
        The name of the logger. Defaults to "CallyticsLogger".
    level : int, optional
        The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Attributes
    ----------
    logger : logging.Logger
        The configured logger instance.
    """

    def __init__(
            self,
            name: Annotated[str, "The name of the logger"] = "CallyticsLogger",
            level: Annotated[int, "The logging level (e.g., logging.INFO)"] = logging.INFO,
    ) -> None:
        """
        Initialize the Logger instance with a specified name and logging level.

        Parameters
        ----------
        name : str, optional
            The name of the logger. Defaults to "CallyticsLogger".
        level : int, optional
            The logging level. Defaults to logging.INFO.

        Returns
        -------
        None
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(
            self,
            message: Annotated[str, "The message to log"],
            level: Annotated[int, "The logging level (e.g., logging.INFO)"] = logging.INFO,
            print_output: Annotated[bool, "Whether to print the message to console"] = True,
    ) -> None:
        """
        Log a message at the specified logging level, with optional console output.

        Parameters
        ----------
        message : str
            The message to log.
        level : int, optional
            The logging level. Defaults to logging.INFO.
        print_output : bool, optional
            Whether to print the message to the console. Defaults to True.

        Returns
        -------
        None

        Examples
        --------
        >>> logger = Logger("ExampleLogger", logging.DEBUG)
        >>> logger.log("This is a debug message", logging.DEBUG)
        This is a debug message
        """
        if print_output:
            print(message)
        self.logger.log(level, message)


class Cleaner:
    """
    A utility class for cleaning up files, directories, or symbolic links at one or multiple specified paths.
    """

    def __init__(self) -> None:
        """
        Initialize the Cleaner class. This method is present for completeness.

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def cleanup(*paths: str) -> None:
        """
        Deletes files, directories, or symbolic links at the specified paths.

        Parameters
        ----------
        *paths : str
            One or more paths to the files or directories to delete.

        Returns
        -------
        None

        Notes
        -----
        - Each path will be checked individually.
        - If the path is a file or symbolic link, it will be deleted.
        - If the path is a directory, the entire directory and its contents will be deleted.
        - If the path does not exist or is neither a file nor a directory, a message will be printed.

        Examples
        --------
        >>> Cleaner.cleanup("/path/to/file", "/path/to/directory")
        File /path/to/file has been deleted.
        Directory /path/to/directory has been deleted.
        """
        for path in paths:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                print(f"File {path} has been deleted.")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Directory {path} has been deleted.")
            else:
                print(f"Path {path} is not a file or directory.")


class Watcher(FileSystemEventHandler):
    """
    A file system event handler that watches a directory for newly created audio files and triggers a callback.

    The Watcher class extends FileSystemEventHandler to monitor a directory for new audio files with specific
    extensions (.mp3, .wav, .flac). When a new file is detected, it invokes an asynchronous callback function,
    allowing users to integrate custom processing logic (e.g., transcription, diarization) immediately after
    the file is created.

    Parameters
    ----------
    callback : callable
        An asynchronous callback function that accepts a single argument (the path to the newly created audio file).
    """

    def __init__(self, callback) -> None:
        """
        Initialize the Watcher with a specified asynchronous callback.

        Parameters
        ----------
        callback : callable
            An async function that will be called with the path of the newly created audio file.

        Returns
        -------
        None
        """
        super().__init__()
        self.callback = callback

    def on_created(self, event) -> None:
        """
        Handle the creation of a new file event.

        If the newly created file is an audio file with supported extensions (.mp3, .wav, .flac),
        this method triggers the asynchronous callback function to process the file.

        Parameters
        ----------
        event : FileSystemEvent
            The event object representing the file system change.

        Returns
        -------
        None
        """
        if not event.is_directory and event.src_path.lower().endswith(('.mp3', '.wav', '.flac')):
            print(f"New audio file detected: {event.src_path}")
            asyncio.run(self.callback(event.src_path))

    @classmethod
    def start_watcher(cls, directory: str, callback) -> None:
        """
        Starts the file system watcher on the specified directory.

        If the directory does not exist, it will be created. The Watcher will monitor the directory for newly
        created audio files and trigger the provided callback function.

        Parameters
        ----------
        directory : str
            The path of the directory to watch.
        callback : callable
            An asynchronous callback function that accepts the path to a newly created audio file.

        Returns
        -------
        None
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory '{directory}' created.")

        observer = Observer()
        event_handler = cls(callback)
        observer.schedule(event_handler, directory, recursive=False)
        observer.start()
        print(f"Watching directory: {directory}")

        import time
        try:
            while True:
                # Senkron bekleme
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


if __name__ == "__main__":
    path_to_file = "sample_file.txt"
    path_to_directory = "sample_directory"

    with open(path_to_file, "w") as file:
        file.write("This is a sample file for testing the Cleaner class.")

    os.makedirs(path_to_directory, exist_ok=True)

    print(f"Attempting to delete file: {path_to_file}")
    Cleaner.cleanup(path_to_file)

    print(f"Attempting to delete directory: {path_to_directory}")
    Cleaner.cleanup(path_to_directory)

    non_existent_path = "non_existent_path"
    print(f"Attempting to delete non-existent path: {non_existent_path}")
    Cleaner.cleanup(non_existent_path)

    Cleaner.cleanup(path_to_file, path_to_directory)
