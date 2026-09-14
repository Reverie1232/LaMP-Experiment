"""
Profile Generator

Generate user profile from selected representative histories.

Input:
    selected histories

Output:
    user profile text

Compatible with LaMP tasks.
"""


import torch

from prompts import (
    get_profile_prompt,
)


class ProfileGenerator:
    def __init__(
        self,
        tokenizer,
        model,
        task_name,
        max_new_tokens=300,
        temperature=0.6,
        top_p=0.9,
        top_k=10,
    ):
        self.tokenizer = tokenizer
        self.model = model

        self.task_name = task_name

        self.max_new_tokens = max_new_tokens

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    #########################################################
    # Build Prompt
    #########################################################

    def build_prompt(
        self,
        selected_histories,
    ):
        """
        Construct profile generation prompt.

        selected_histories:

        [
            {
                "history": "...",
                "score":9.2
            }
        ]

        """

        history_texts = []


        for item in selected_histories:

            history_texts.append(
                item["history"]
            )


        history_string = "\n".join(
            history_texts
        )


        prompt = get_profile_prompt(
            self.task_name,
            history_string,
        )


        return prompt



    #########################################################
    # Generate once
    #########################################################

    @torch.no_grad()
    def generate_once(
        self,
        prompt,
    ):


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        print(f"prompt:{prompt}")
        inputs = inputs.to(
            self.model.device
        )
        print(inputs)

        with torch.autocast(
            device_type="cuda"
        ):


            outputs = self.model.generate(

                **inputs,

                do_sample=True,

                temperature=self.temperature,

                top_p=self.top_p,

                top_k=self.top_k,

                max_new_tokens=self.max_new_tokens,

                eos_token_id=
                self.tokenizer.eos_token_id,

            )


        output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )


        # remove prompt

        output = output.replace(
            prompt,
            ""
        ).strip()

        print(f"output:{output}")
        return output



    #########################################################
    # Main Interface
    #########################################################

    def generate_profile(
        self,
        selected_histories,
    ):

        prompt = self.build_prompt(
            selected_histories
        )


        profile = self.generate_once(
            prompt
        )


        return profile


#############################################################
# Test
#############################################################

if __name__=="__main__":

    print(
        "ProfileGenerator module loaded."
    )