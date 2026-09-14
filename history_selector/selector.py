"""
History Selector

Select representative histories according to
LLM importance score.

Supported:
1. Top-K ranking
2. Score threshold selection
3. MMR diversity selection (optional)
"""

import numpy as np
from typing import List, Dict


class HistorySelector:


    def __init__(
        self,
        method="topk",
        topk=10,
        lambda_mmr=0.7,
    ):
        """
        Args:

            method:
                topk
                mmr

            topk:
                number of selected histories

            lambda_mmr:
                trade-off between relevance and diversity
        """

        self.method = method
        self.topk = topk
        self.lambda_mmr = lambda_mmr



    ############################################################
    # Basic Top-K Selection
    ############################################################

    def select_topk(
        self,
        ranked_histories: List[Dict],
    ):
        """
        Select highest score histories.

        Input:

        [
            {
                "history": "...",
                "score": 9.2,
                "reason": "..."
            }
        ]

        """

        ranked_histories = sorted(
            ranked_histories,
            key=lambda x: x["score"],
            reverse=True,
        )


        return ranked_histories[:self.topk]



    ############################################################
    # Score Threshold Selection
    ############################################################

    def select_threshold(
        self,
        ranked_histories,
        threshold=8.0,
    ):
        """
        Select histories with score higher than threshold.
        """

        selected = []


        for item in ranked_histories:

            if item["score"] >= threshold:

                selected.append(item)


        return selected[:self.topk]



    ############################################################
    # MMR Selection
    ############################################################

    def select_mmr(
        self,
        ranked_histories,
    ):
        """
        Maximal Marginal Relevance selection.


        score =
        
        lambda * importance
        -
        (1-lambda) * redundancy


        Need each item contains:

        embedding


        Example:

        {
            history:"",
            score:8.5,
            embedding:[...]
        }

        """


        if len(ranked_histories) <= self.topk:

            return ranked_histories



        selected = []

        candidates = list(
            ranked_histories
        )


        # normalize score

        scores = [
            x["score"]
            for x in candidates
        ]


        max_score=max(scores)

        min_score=min(scores)


        for item in candidates:

            if max_score == min_score:

                item["norm_score"] = 1.0

            else:

                item["norm_score"] = (
                    item["score"]-min_score
                ) / (
                    max_score-min_score
                )



        ##################################################
        # first select highest score
        ##################################################

        first = max(
            candidates,
            key=lambda x:x["norm_score"]
        )

        selected.append(first)

        candidates.remove(first)



        ##################################################
        # iterative MMR
        ##################################################

        while len(selected) < self.topk:

            best_item=None

            best_value=-1e9


            for candidate in candidates:


                relevance = candidate[
                    "norm_score"
                ]


                redundancy = 0


                for chosen in selected:


                    sim = self.cosine_similarity(

                        candidate["embedding"],

                        chosen["embedding"]

                    )


                    redundancy=max(
                        redundancy,
                        sim
                    )


                mmr_score = (

                    self.lambda_mmr
                    *
                    relevance

                    -

                    (1-self.lambda_mmr)
                    *
                    redundancy

                )


                if mmr_score > best_value:

                    best_value=mmr_score

                    best_item=candidate



            selected.append(best_item)

            candidates.remove(best_item)


        return selected



    ############################################################
    # Cosine Similarity
    ############################################################

    @staticmethod
    def cosine_similarity(
        a,
        b,
    ):

        a=np.array(a)

        b=np.array(b)


        denom=(

            np.linalg.norm(a)

            *

            np.linalg.norm(b)

        )


        if denom==0:

            return 0


        return float(
            np.dot(a,b)
            /
            denom
        )



    ############################################################
    # Unified Interface
    ############################################################

    def select(
        self,
        ranked_histories,
    ):

        if self.method=="topk":

            return self.select_topk(
                ranked_histories
            )


        elif self.method=="threshold":

            return self.select_threshold(
                ranked_histories
            )


        elif self.method=="mmr":

            return self.select_mmr(
                ranked_histories
            )


        else:

            raise ValueError(
                f"Unknown selection method {self.method}"
            )



############################################################
# Test
############################################################


if __name__=="__main__":


    selector=HistorySelector(
        method="topk",
        topk=2,
    )


    histories=[

        {
            "history":
            "movie: Interstellar",

            "score":9.5,

            "reason":
            "science fiction preference"
        },


        {
            "history":
            "movie: Titanic",

            "score":8.2,

            "reason":
            "romance preference"
        },


        {
            "history":
            "movie: Avatar",

            "score":7.8,

            "reason":
            "fantasy preference"
        }

    ]


    result=selector.select(
        histories
    )


    for x in result:

        print(
            x
        )