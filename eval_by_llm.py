import pandas as pd
from openai import OpenAI
import os
import re
from tqdm import tqdm
tqdm.pandas()

# Evaluation prompt template
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an conversation. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{conversation}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source conversation. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
1: Poor. 
   - The summary includes very little relevant content from the source conversation.
   - It fails to capture essential details and main points.
   - Contains a significant amount of irrelevant or redundant information.
2: Limited. 
   - The summary includes some relevant content but overlooks key details and main points from the source.
   - There may be redundancies or minor irrelevant information present.
   - The selection of important content is inadequate.
3: Moderate. 
   - The summary includes a moderate amount of relevant content from the source conversation.
   - It captures the main points but may lack depth in some areas.
   - There might be occasional redundancies or minor irrelevant information.
4: Good. 
   - The summary includes a good amount of relevant content from the source conversation.
   - It effectively captures and conveys the main points and key details.
   - Redundancies and irrelevant information are minimal or non-disruptive.
5: Excellent.  
   - The summary demonstrates excellent relevance by selecting and including all important content from the source conversation.
   - It provides a clear and concise representation of the main points and key details.
   - There are no redundancies or excess information; every sentence contributes to the overall understanding.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source conversation carefully.
2. Compare the summary to the source conversation and identify the main points of the conversation.
3. Assess how well the summary covers the main points of the conversation, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
The summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic.
1: Poor. 
   - The summary is highly disorganized and lacks any logical flow. 
   - Sentences are disconnected, making it difficult to follow the conversation. 
   - Information is scattered, and the summary fails to build a coherent narrative.
2: Limited. 
   - The summary has some organization, but it is still challenging to follow.
   - There is a loose attempt to group related sentences, but transitions are weak.
   - The summary does not effectively build a coherent body of information.
3: Moderate. 
   - The summary demonstrates a moderate level of organization.
   - Sentences are somewhat connected, making it somewhat easier to follow.
   - There is an attempt to build a coherent narrative, but it could be improved.
4: Good. 
   - The summary is well-structured and organized.
   - Sentences are logically connected, leading to a clear and coherent narrative.
   - Information is presented in a way that progressively builds upon the previous sentences.
5: Excellent.
   - The summary is highly organized and exceptionally coherent.
   - Sentences flow seamlessly, creating a smooth and logical progression of information.
   - It effectively builds a cohesive and complete narrative that is easy to follow and understand.
"""

COHERENCE_SCORE_STEPS = """
1. Read the conversation carefully and identify the main topic and key points.
2. Read the summary and compare it to the conversation. Check if the summary covers the main topic and key points of the conversation,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source conversation. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
1: Poor.
   - The summary contains numerous factual inaccuracies that do not align with the source conversation.
   - It includes hallucinated or completely false information.
   - There is a significant disconnect between the summary and the original conversation.
2: Limited. 
   - The summary contains several factual inaccuracies or inconsistencies with the source conversation.
   - It may include some hallucinated facts or details that are not supported by the original discussion.
   - The alignment between the summary and source is weak.
3. Moderate. 
   - The summary demonstrates moderate consistency with the source conversation.
   - While it generally aligns with the original discussion, there may be occasional factual inaccuracies or minor discrepancies.
   - Some statements in the summary might not be fully supported by the source.
4. Good. 
   - The summary is factually consistent with the source conversation.
   - It accurately reflects the main points and details of the original discussion.
   - Factual inaccuracies or hallucinated facts are minimal or non-existent.
5. Excellent. 
   - The summary exhibits excellent consistency with the source conversation.
   - It precisely and faithfully represents the factual content and details of the original discussion.
   - There are no factual inaccuracies or hallucinated facts, and the summary aligns perfectly with the source.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the conversation carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the conversation. Check if the summary contains any factual errors that are not supported by the conversation.
3. Assign a score for consistency based on the Evaluation Criteria, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-5): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. 
   - The summary is riddled with grammatical errors, misspellings, and punctuation mistakes.
   - It is challenging to understand due to the numerous language issues.
   - Word choice and sentence structure are highly problematic, making the summary incoherent.
2: Limited. 
   - The summary contains several grammatical errors, misspellings, and punctuation issues.
   - Language problems affect comprehension to a significant degree.
   - Word choice and sentence structure are subpar but not completely incomprehensible.
3: Moderate.
   - The summary exhibits moderate fluency but still contains some noticeable language issues.
   - While there are some grammatical errors and misspellings, they do not severely hinder understanding.
   - Word choice and sentence structure are generally acceptable but may need improvement.
4: Good. 
   - The summary is fluently written with minimal grammatical errors, misspellings, or punctuation mistakes.
   - Language-related issues do not significantly impact comprehension.
   - Word choice and sentence structure are well-crafted and contribute to overall clarity.
5: Excellent.  
   - The summary demonstrates excellent fluency in terms of grammar, spelling, punctuation, word choice, and sentence structure.
   - It is impeccably written and free of language-related issues.
   - Word choice and sentence structure are highly polished, enhancing the overall quality and readability of the summary.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency. Assign a fluency score from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def get_geval_score(
    criteria: str, steps: str, conversation: str, summary: str, metric_name: str, verbose=False
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        conversation=conversation,
        summary=summary,
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    result = response.choices[0].message.content

    if result and re.findall("[0-9]+",result):
      score_num = int(re.findall("[0-9]+",result)[0])
    else:
      score_num = pd.NA

    if verbose:
        print('\n{0}:{1}'.format(metric_name, score_num))
    return score_num


evaluation_metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}


df = pd.read_csv('results/falcon_7b_LoRA_r16_dialogue_summarization_12_13_2023_results.csv')

for metric, (criteria, steps) in evaluation_metrics.items():
    col_name_human = metric+'_human_summary'
    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print("Generating human baseline summarization score for {0}".format(metric))
    df[col_name_human] = df.progress_apply(lambda x: get_geval_score(criteria=criteria, steps=steps, conversation=x.inputs, 
                                                            summary=x.summary_human_baseline, 
                                                            metric_name=metric), axis=1)
    
    print("Human Summary Score for {0} : {1}".format(metric, df[col_name_human].describe()))

    print(dash_line)
    col_name_model = metric+'_peft_model_summary'
    print("Generating PEFT baseline summarization score for {0}".format(metric))
    df[col_name_model] = df.progress_apply(lambda x: get_geval_score(criteria=criteria, steps=steps, conversation=x.inputs, 
                                                            summary=x.summary_peft_baseline,
                                                            metric_name=metric), axis=1)
    print("PEFT Model Summary Score for {0} : {1}".format(metric, df[col_name_model].describe()))

df.to_csv('results/falcon_7b_LoRA_r16_dialogue_summarization_12_13_2023_results_eval.csv')





