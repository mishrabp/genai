from utility.llm import LLM
from langchain_core.output_parsers import StrOutputParser

import json
import random
import session_info

import pandas as pd
import numpy as np

from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = StrOutputParser()

client = LLM.get_client()
chain = client | parser
# if client :
#     print(chain.invoke("what is generative ai?"))

def evaluate_prompt(prompt, gold_examples, user_message_template,samples_to_output = 10):

    """
    Return the accuracy score for predictions on gold examples.
    For each example, we make a prediction using the prompt. Gold labels and
    model predictions are aggregated into lists and compared to compute the
    accuracy.

    Args:
        prompt (List): list of messages in the Open AI prompt format
        gold_examples (str): JSON string with list of gold examples
        user_message_template (str): string with a placeholder for product description
        samples_to_output (int): number of sample predictions and ground truths to print

    Output:
        accuracy (float): Accuracy computed by comparing model predictions
                                with ground truth
    """

    count =0
    model_predictions, ground_truths = [], []

    for example in json.loads(gold_examples):
        gold_input = example['Product Description']
        user_input = [
            {
                'role':'user',
                'content': user_message_template.format(product_description=gold_input)
            }
        ]

        try:
            # response = client.chat.completions.create(
            #     model=deployment_name,
            #     messages=prompt+user_input,
            #     temperature=0, # <- Note the low temperature
            #     max_tokens=4 # <- Note how we restrict the output to not more than 4 tokens
            # )

            # prediction = response.choices[0].message.content

            prediction=chain.invoke(prompt+user_input,
                config={"temperature": 0.5, "max_tokens": 4})

            # print(prediction) #uncomment to see LLM response or to debug
            model_predictions.append(prediction.strip().lower()) # <- removes extraneous white space and lowercases output
            ground_truths.append(example['Category'].strip().lower())

            if count < samples_to_output:
              count += 1
              print("Product Description: \n", example['Product Description'],"\n")
              print("Original label: \n", example['Category'],"\n")
              print("Predicted label: \n", prediction)
              print("====================================================")

        except Exception as e:
            print(e)
            continue

        accuracy = accuracy_score(ground_truths, model_predictions)

    return accuracy

def evaluate_zero_shot_promt(gold_examples):
    user_message_template = """```{product_description}```"""
    zero_shot_system_message = """
        Classify the following product desciption presented in the input into one of the following categories.
        Categories - ['Hair Care', 'Skin Care']
        Product description will be delimited by triple backticks in the input.
        Answer only 'Hair Care' or 'Skin Care'. Nothing Else. Do not explain your answer.
    """

    zero_shot_prompt = [{'role':'system', 'content': zero_shot_system_message}]

    token_size = LLM.num_tokens_from_messages(zero_shot_prompt)

    accuracy = evaluate_prompt(zero_shot_prompt, gold_examples, user_message_template)


def create_examples(dataset, n=4):

    """
    Return a JSON list of randomized examples of size 2n with two classes.
    Create subsets of each class, choose random samples from the subsets,
    merge and randomize the order of samples in the merged list.
    Each run of this function creates a different random sample of examples
    chosen from the training data.

    Args:
        dataset (DataFrame): A DataFrame with examples (text + label)
        n (int): number of examples of each class to be selected

    Output:
        randomized_examples (JSON): A JSON with examples in random order
    """

    hc_reviews = (examples_df.Category == 'Hair Care')
    sc_reviews = (examples_df.Category == 'Skin Care')

    cols_to_select = ["Product Description","Category"]
    hc_examples = examples_df.loc[hc_reviews, cols_to_select].sample(n)
    sc_examples = examples_df.loc[sc_reviews, cols_to_select].sample(n)

    examples = pd.concat([hc_examples,sc_examples])
    # sampling without replacement is equivalent to random shuffling
    randomized_examples = examples.sample(2*n, replace=False)

    return randomized_examples.to_json(orient='records')

def create_prompt(system_message, examples, user_message_template):

    """
    Return a prompt message in the format expected by the Open AI API.
    Loop through the examples and parse them as user message and assistant
    message.

    Args:
        system_message (str): system message with instructions for classification
        examples (str): JSON string with list of examples
        user_message_template (str): string with a placeholder for description

    Output:
        few_shot_prompt (List): A list of dictionaries in the Open AI prompt format
    """

    few_shot_prompt = [{'role':'system', 'content': system_message}]

    for example in json.loads(examples):
        example_description = example['Product Description']
        example_category = example['Category']

        few_shot_prompt.append(
            {
                'role': 'user',
                'content': user_message_template.format(
                    product_description=example_description
                )
            }
        )

        few_shot_prompt.append(
            {'role': 'assistant', 'content': f"{example_category}"}
        )

    return few_shot_prompt

if __name__ == "__main__":

    ## Prepare data for example and testing
    data = pd.read_csv("/mnt/c/Users/msuni/genai/_data/auto-labelling.csv")
    examples_df, gold_df = train_test_split(
        data, #<- the full dataset
        test_size=0.8, #<- 80% random sample selected for gold examples
        random_state=42, #<- ensures that the splits are the same for every session
        stratify=data['Category'] #<- ensures equal distribution of labels
    )

    gold_examples = (
        gold_df.to_json(orient='records')
    )

    evaluate_zero_shot_promt(gold_examples)


    # examples = create_examples(examples_df, 2)
    # few_shot_prompt = create_prompt(
    #     few_shot_system_message,
    #     examples,
    #     user_message_template
    # )





