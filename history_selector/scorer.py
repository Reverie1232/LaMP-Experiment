import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parser import self_consistency
from prompts import get_history_scoring_prompt


class HistoryScorer:

    def __init__(
        self,
        tokenizer,
        model,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        max_new_tokens=128,
        sample_times=5,
    ):

        self.sample_times = sample_times

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        self.tokenizer = tokenizer

        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "[PAD]"

        self.model = model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.use_cache = True

    #####################################################################
    # generate one output
    #####################################################################

    @torch.no_grad()
    def generate_once(self, prompt):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        inputs = inputs.to(self.model.device)

        with torch.autocast(device_type="cuda"):

            outputs = self.model.generate(

                **inputs,

                do_sample=True,

                temperature=self.temperature,

                top_p=self.top_p,

                top_k=self.top_k,

                max_new_tokens=self.max_new_tokens,

                eos_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        # print(f"output:{output}")
        output = output.replace(prompt, "").strip()
        
        return output

    #####################################################################
    # self consistency
    #####################################################################

    def score_history(self, history_text):

        prompt = get_history_scoring_prompt(history_text)

        outputs = []

        for _ in range(self.sample_times):

            out = self.generate_once(prompt)

            outputs.append(out)

        result = self_consistency(outputs)

        return result

    #####################################################################
    # batch scoring
    #####################################################################

    def score_histories(self, history_list):

        results = []

        for history in history_list:

            result = self.score_history(history)

            results.append(result)

        return results

    #####################################################################
    # rank
    #####################################################################

    def rank_histories(self, history_list):

        results = self.score_histories(history_list)

        idx = list(range(len(history_list)))

        idx = sorted(
            idx,
            key=lambda x: results[x]["score"],
            reverse=True,
        )

        ranked_history = []

        for i in idx:

            ranked_history.append({

                "history": history_list[i],

                "score": results[i]["score"],

                "majority_score": results[i]["majority_score"],

                "reason": results[i]["reason"],

                "all_scores": results[i]["all_scores"],

                "all_reasons": results[i]["all_reasons"],

            })

        return ranked_history

    #####################################################################
    # top-k
    #####################################################################

    def select_topk(self, history_list, k=10):

        ranked = self.rank_histories(history_list)

        return ranked[:k]


if __name__ == "__main__":

    scorer = HistoryScorer(
        sample_times=5,
    )

    history = [
        "description: Interstellar tag: sci-fi",
        "description: Titanic tag: romance",
        "description: Avengers tag: action",
    ]

    result = scorer.select_topk(
        history,
        k=2,
    )

    for r in result:

        print("=" * 60)

        print(r["history"])

        print("Score:", r["score"])

        print("Reason:", r["reason"])