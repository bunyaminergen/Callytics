# Standard library imports
from typing import Annotated, Dict, Any, List


class Annotator:
    """
    A class to annotate a structured sentiment model (SSM) with various
    attributes such as sentiment, profanity, summary, conflict, and topic.

    Parameters
    ----------
    ssm : list of dict
        A list of dictionaries representing the structured sentiment model.

    Attributes
    ----------
    ssm : list of dict
        The structured sentiment model to be annotated.
    global_summary : str
        The global summary of the annotations.
    global_conflict : bool
        The global conflict status of the annotations.
    global_topic : str
        The global topic of the annotations.
    """

    def __init__(self, ssm: Annotated[List[Dict[str, Any]], "Structured Sentiment Model"]):
        """
        Initializes the Annotator class with the provided SSM.

        Parameters
        ----------
        ssm : list of dict
            A list of dictionaries representing the structured sentiment model.
        """
        self.ssm = ssm
        self.global_summary = ""
        self.global_conflict = False
        self.global_topic = "Unknown"

    def add_sentiment(
            self,
            sentiment_results: Annotated[Dict[str, Any], "Sentiment analysis results"]
    ):
        """
        Adds sentiment data to the SSM.

        Parameters
        ----------
        sentiment_results : dict
            A dictionary containing sentiment analysis results, including
            a "sentiments" key with a list of sentiment dictionaries.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> results = {"sentiments": [{"index": 0, "sentiment": "Positive"}]}
        >>> annotator.add_sentiment(sentiment_results)
        """
        if len(sentiment_results["sentiments"]) != len(self.ssm):
            print(f"Mismatch: SSM Length = {len(self.ssm)}, "
                  f"Sentiments Length = {len(sentiment_results['sentiments'])}")
            print("Adjusting to match lengths...")

        if len(sentiment_results["sentiments"]) < len(self.ssm):
            for idx in range(len(sentiment_results["sentiments"]), len(self.ssm)):
                sentiment_results["sentiments"].append({"index": idx, "sentiment": "Neutral"})

        elif len(sentiment_results["sentiments"]) > len(self.ssm):
            sentiment_results["sentiments"] = sentiment_results["sentiments"][:len(self.ssm)]

        for sentiment_data in sentiment_results["sentiments"]:
            idx = sentiment_data["index"]
            if idx < len(self.ssm):
                self.ssm[idx]["sentiment"] = sentiment_data["sentiment"]
            else:
                print(f"Skipping sentiment data at index {idx}, out of range.")

    def add_profanity(
            self,
            profane_results: Annotated[Dict[str, Any], "Profanity detection results"]
    ) -> List[Dict[str, Any]]:
        """
        Adds profanity data to the SSM.

        Parameters
        ----------
        profane_results : dict
            A dictionary containing profanity detection results, including
            a "profanity" key with a list of profanity dictionaries.

        Returns
        -------
        list of dict
            The updated SSM with profanity annotations.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> results = {"profanity": [{"index": 0, "profane": True}]}
        >>> annotator.add_profanity(profane_results)
        """
        if "profanity" not in profane_results:
            print("Warning: 'profanity' key is missing in profane_results.")
            return self.ssm

        if len(profane_results["profanity"]) != len(self.ssm):
            print(f"Mismatch: SSM Length = {len(self.ssm)}, "
                  f"Profanity Length = {len(profane_results['profanity'])}")
            print("Adjusting to match lengths...")

        if len(profane_results["profanity"]) < len(self.ssm):
            for idx in range(len(profane_results["profanity"]), len(self.ssm)):
                profane_results["profanity"].append({"index": idx, "profane": False})

        elif len(profane_results["profanity"]) > len(self.ssm):
            profane_results["profanity"] = profane_results["profanity"][:len(self.ssm)]

        for profanity_data in profane_results["profanity"]:
            idx = profanity_data["index"]
            if idx < len(self.ssm):
                self.ssm[idx]["profane"] = profanity_data["profane"]
            else:
                print(f"Skipping profanity data at index {idx}, out of range.")

        return self.ssm

    def add_summary(
            self,
            summary_result: Annotated[Dict[str, str], "Summary results"]
    ) -> Dict[str, Any]:
        """
        Adds a global summary to the annotations.

        Parameters
        ----------
        summary_result : dict
            A dictionary containing a "summary" key with the summary text.

        Returns
        -------
        dict
            The updated SSM and global summary.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> result = {"summary": "This is a summary."}
        >>> annotator.add_summary(summary_result)
        """
        if not summary_result or "summary" not in summary_result:
            print("Warning: 'summary' key is missing in summary_result.")
            return {"ssm": self.ssm, "summary": self.global_summary}

        self.global_summary = summary_result["summary"]
        return {"ssm": self.ssm, "summary": self.global_summary}

    def add_conflict(
            self,
            conflict_result: Annotated[Dict[str, bool], "Conflict detection results"]
    ) -> Dict[str, Any]:
        """
        Adds a global conflict status to the annotations.

        Parameters
        ----------
        conflict_result : dict
            A dictionary containing a "conflict" key with a boolean value.

        Returns
        -------
        dict
            The updated SSM and global conflict status.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> result = {"conflict": True}
        >>> annotator.add_conflict(conflict_result)
        """
        if not conflict_result or "conflict" not in conflict_result:
            print("Warning: 'conflict' key is missing in conflict_result.")
            return {"ssm": self.ssm, "conflict": self.global_conflict}

        self.global_conflict = conflict_result["conflict"]
        return {"ssm": self.ssm, "conflict": self.global_conflict}

    def add_topic(
            self,
            topic_result: Annotated[Dict[str, str], "Topic detection results"]
    ) -> Dict[str, Any]:
        """
        Adds a global topic to the annotations.

        Parameters
        ----------
        topic_result : dict
            A dictionary containing a "topic" key with the topic name.

        Returns
        -------
        dict
            The updated SSM and global topic.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> result = {"topic": "Technology"}
        >>> annotator.add_topic(topic_result)
        """
        if not topic_result or "topic" not in topic_result:
            print("Warning: 'topic' key is missing in topic_result.")
            return {"ssm": self.ssm, "topic": self.global_topic}

        self.global_topic = topic_result["topic"]
        return {"ssm": self.ssm, "topic": self.global_topic}

    def finalize(self) -> Dict[str, Any]:
        """
        Finalizes the annotations by returning the updated SSM along with
        global annotations for summary, conflict, and topic.

        Returns
        -------
        dict
            A dictionary containing the updated SSM and global annotations.

        Examples
        --------
        >>> annotator = Annotator([{"text": "example"}])
        >>> annotator.finalize()
        {'ssm': [{'text': 'example'}], 'summary': '', 'conflict': False, 'topic': 'Unknown'}
        """
        return {
            "ssm": self.ssm,
            "summary": self.global_summary,
            "conflict": self.global_conflict,
            "topic": self.global_topic
        }
