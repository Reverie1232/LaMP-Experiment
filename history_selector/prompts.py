"""
Prompt Library

History Selector
Profile Generator

Compatible with all LaMP tasks.
"""

###########################################################
# Universal History Scoring Prompt
###########################################################

HISTORY_SCORING_PROMPT = """
You are an expert in personalized user modeling.

Your task is to evaluate whether ONE user behavior
is useful for building a compact and accurate user profile.

A representative behavior should satisfy MOST of the following:

1. Reveals a stable long-term preference.
2. Reflects the user's habitual decision pattern.
3. Contains distinctive personalized information.
4. Is not merely accidental or noisy.
5. Helps predict future user behavior.

Please evaluate the following behavior.

Behavior:
------------------------
{history}
------------------------

Please answer strictly in the following format.

Importance Score: <1-10>

Reason:
<one concise sentence>

Do NOT output anything else.
"""


###########################################################
# Generic Profile Prompt
###########################################################

PROFILE_PROMPT = """
You are an expert in user profiling.

Below are several representative user behaviors that have already
been selected because they best characterize this user.

Your task is NOT to summarize every behavior.

Instead, infer the user's stable characteristics, including:

- Long-term interests
- Decision patterns
- Personal preferences
- Writing style (if applicable)
- Other consistent behavioral traits

Representative Behaviors:
------------------------
{history}
------------------------

Only output the user profile.
"""


###########################################################
# Reflection Prompt (Future Extension)
###########################################################

PROFILE_REFLECTION_PROMPT = """
You are reviewing a generated user profile.

User Profile:
------------------------
{profile}
------------------------

Representative Behaviors:
------------------------
{history}
------------------------

Please determine whether every statement in the profile
is supported by the behaviors.

If unsupported statements exist,
rewrite the profile.

Only output the revised profile.
"""


###########################################################
# Task-specific Profile Prompt
###########################################################

TASK_PROFILE_PROMPTS = {

    "movie_tagging":
"""
Infer the user's movie tagging preference from the representative behaviors.

Focus on:

- favorite genres
- tagging habits
- labeling strategy
- stable interests

Representative Behaviors:

{history}

Only output the profile.
""",

    "citation":
"""
Write a summary, in English, 
of the research interests and topics of a researcher who has published the following papers. 
Only generate the summary, no other text. User History: {history} Answer:
""",

    "news_categorize":
"""
Infer the journalist's writing preference
from representative articles.

Focus on:

- preferred news topics
- writing interests
- category preference

Representative Behaviors:

{history}

Only output the profile.
""",

    "news_headline":
"""
Infer this author's headline writing style.

Focus on:

- wording
- sentence pattern
- headline length
- common expressions

Representative Behaviors:

{history}

Only output the profile.
""",

    "product_rating":
"""
Infer the user's rating habit.

Focus on:

- strictness
- scoring tendency
- positive review pattern
- negative review pattern

Representative Behaviors:

{history}

Only output the profile.
""",

    "scholarly_title":
"""
Infer the author's title writing style.

Focus on:

- title template
- wording
- common structure
- terminology

Representative Behaviors:

{history}

Only output the profile.
""",

    "tweet_paraphrase":
"""
Infer this user's writing style.

Focus on:

- wording
- punctuation
- capitalization
- tone
- emoji usage
- sentence length

Representative Behaviors:

{history}

Only output the profile.
"""
}


###########################################################
# History Formatting
###########################################################

RETRIEVAL_HISTORY = {

    "movie_tagging":
"description: {description}\ntag: {tag}",

    "citation":
"paper title: {title}\nreference: {citation}",

    "news_categorize":
"article: {text}\ncategory: {category}",

    "news_headline":
"article: {text}\nheadline: {title}",

    "product_rating":
"review: {text}\nscore: {score}",

    "scholarly_title":
"abstract: {abstract}\ntitle: {title}",

    "tweet_paraphrase":
"tweet: {text}",
}


###########################################################
# Helper Functions
###########################################################

def get_history_scoring_prompt(history):

    return HISTORY_SCORING_PROMPT.format(
        history=history
    )


def get_profile_prompt(task_name, history):

    if task_name in TASK_PROFILE_PROMPTS:

        return TASK_PROFILE_PROMPTS[task_name].format(
            history=history
        )

    return PROFILE_PROMPT.format(
        history=history
    )


def get_reflection_prompt(profile, history):

    return PROFILE_REFLECTION_PROMPT.format(
        profile=profile,
        history=history,
    )


def format_history(task_name, history_dict):

    return RETRIEVAL_HISTORY[task_name].format(
        **history_dict
    )