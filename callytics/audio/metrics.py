# Standard library imports
import math
from typing import Annotated, List, Dict

# Related third-party imports
import numpy as np


class SilenceStats:
    """
    A class to compute and analyze statistics for silence durations
    between speech segments.

    This class provides methods to compute common statistical metrics
    (mean, median, standard deviation, interquartile range) and thresholds
    based on silence durations.

    Attributes
    ----------
    silence_durations : List[float]
        A sorted list of silence durations.

    Methods
    -------
    from_segments(segments)
        Class method to create a SilenceStats instance from speech segments.
    median()
        Compute the median silence duration.
    mean()
        Compute the mean silence duration.
    std()
        Compute the standard deviation of silence durations.
    iqr()
        Compute the interquartile range (IQR) of silence durations.
    threshold_std(factor=0.95)
        Compute threshold based on standard deviation.
    threshold_median_iqr(factor=1.5)
        Compute threshold based on median + IQR.
    total_silence_above_threshold(threshold)
        Compute total silence above a given threshold.
    """

    def __init__(self, silence_durations: Annotated[List[float], "List of silence durations"]):
        """
        Initialize the SilenceStats class with a list of silence durations.

        Parameters
        ----------
        silence_durations : List[float]
            List of silence durations (non-negative values).
        """
        if not all(isinstance(x, (int, float)) and x >= 0 for x in silence_durations):
            raise ValueError("silence_durations must be a list of non-negative numbers.")
        self.silence_durations = sorted(silence_durations)

    @classmethod
    def from_segments(cls, segments: Annotated[List[Dict], "List of speech segments"]) -> "SilenceStats":
        """
        Create a SilenceStats instance from a list of speech segments.

        Parameters
        ----------
        segments : List[Dict]
            List of speech segments, where each segment contains 'start_time'
            and 'end_time' keys.

        Returns
        -------
        SilenceStats
            A SilenceStats instance with computed silence durations.

        Examples
        --------
        >>> segment = [{"start_time": 0, "end_time": 5}, {"start_time": 10, "end_time": 15}]
        >>> stat = SilenceStats.from_segments(segments)
        >>> stat.silence_durations
        [5]
        """
        segments_sorted = sorted(segments, key=lambda x: x['start_time'])
        durations = [
            segments_sorted[i + 1]['start_time'] - segments_sorted[i]['end_time']
            for i in range(len(segments_sorted) - 1)
            if (segments_sorted[i + 1]['start_time'] - segments_sorted[i]['end_time']) > 0
        ]
        return cls(durations)

    def median(self) -> Annotated[float, "Median of silence durations"]:
        """
        Compute the median silence duration.

        Returns
        -------
        float
            The median of the silence durations.
        """
        n = len(self.silence_durations)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 0:
            return (self.silence_durations[mid - 1] + self.silence_durations[mid]) / 2
        return self.silence_durations[mid]

    def mean(self) -> Annotated[float, "Mean of silence durations"]:
        """
        Compute the mean silence duration.

        Returns
        -------
        float
            The mean of the silence durations.
        """
        return sum(self.silence_durations) / len(self.silence_durations) if self.silence_durations else 0.0

    def std(self) -> Annotated[float, "Standard deviation of silence durations"]:
        """
        Compute the standard deviation of silence durations.

        Returns
        -------
        float
            The standard deviation of the silence durations.
        """
        n = len(self.silence_durations)
        if n == 0:
            return 0.0
        mu = self.mean()
        var = sum((x - mu) ** 2 for x in self.silence_durations) / n
        return math.sqrt(var)

    def iqr(self) -> Annotated[float, "Interquartile range (IQR) of silence durations"]:
        """
        Compute the Interquartile Range (IQR).

        Returns
        -------
        float
            The IQR of the silence durations.
        """
        if not self.silence_durations:
            return 0.0
        q1 = np.percentile(self.silence_durations, 25)
        q3 = np.percentile(self.silence_durations, 75)
        return q3 - q1

    def threshold_std(self, factor: Annotated[float, "Scaling factor for std threshold"] = 0.95) -> float:
        """
        Compute the threshold based on standard deviation.

        Parameters
        ----------
        factor : float, optional
            A scaling factor for the standard deviation, by default 0.95.

        Returns
        -------
        float
            Threshold based on standard deviation.
        """
        return self.std() * factor

    def threshold_median_iqr(self, factor: Annotated[float, "Scaling factor for IQR"] = 1.5) -> float:
        """
        Compute the threshold based on median and IQR.

        Parameters
        ----------
        factor : float, optional
            A scaling factor for the IQR, by default 1.5.

        Returns
        -------
        float
            Threshold based on median and IQR.
        """
        return self.median() + (self.iqr() * factor)

    def total_silence_above_threshold(
            self, threshold: Annotated[float, "Threshold value for silence"]
    ) -> Annotated[float, "Total silence above the threshold"]:
        """
        Compute the total silence above the given threshold.

        Parameters
        ----------
        threshold : float
            The threshold value to compare silence durations.

        Returns
        -------
        float
            Total silence duration above the threshold.
        """
        return sum(s for s in self.silence_durations if s >= threshold)


if __name__ == "__main__":
    final_ssm = {
        'ssm': [
            {'speaker': 'Customer', 'start_time': 8500, 'end_time': 9760, 'text': 'Hey, G-Chance, this is Jennifer. ',
             'index': 0, 'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 10660, 'end_time': 11560, 'text': 'Yes, hi, Jennifer. ', 'index': 1,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 11620, 'end_time': 12380, 'text': "Good afternoon, ma'am. ", 'index': 2,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 83880, 'end_time': 85460, 'text': 'Okay. ', 'index': 24,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 85500, 'end_time': 85620, 'text': 'Yeah. ', 'index': 25,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 86400, 'end_time': 90320,
             'text': "So I'll be sending this shipping documents right after this call. ", 'index': 26,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'CSR', 'start_time': 90400, 'end_time': 91160, 'text': 'Thank you so much. ', 'index': 27,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'Customer', 'start_time': 92060, 'end_time': 92680, 'text': 'Okay, thank you. ', 'index': 28,
             'sentiment': 'Neutral', 'profane': False},
            {'speaker': 'Customer', 'start_time': 93880, 'end_time': 98220, 'text': 'All right, bye-bye. ', 'index': 29,
             'sentiment': 'Neutral', 'profane': False}
        ],
        'summary': 'Gabby from Transplace AP Team called Jennifer to request copies of a carrier invoice, bill of '
                   'lading, and proof of delivery, and Jennifer provided her email for Gabby to send the shipping '
                   'documents.',
        'conflict': False,
        'topic': 'Invoice and Shipping Documents Request'
    }

    stats = SilenceStats.from_segments(final_ssm['ssm'])

    print("Mean:", stats.mean())
    print("Median:", stats.median())
    print("Std Dev:", stats.std())
    print("IQR:", stats.iqr())

    t_std = stats.threshold_std(factor=0.97)
    t_median_iqr = stats.threshold_median_iqr(factor=1.49)
    print("Threshold (std-based):", t_std)
    print("Threshold (median+IQR):", t_median_iqr)

    print("Total silence (std-based):", stats.total_silence_above_threshold(t_std))
    print("Total silence (median+IQR-based):", stats.total_silence_above_threshold(t_median_iqr))
    final_ssm["silence"] = t_std
    print(final_ssm)
