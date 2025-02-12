############## PROBLEM STATEMENT
### We have received 1000s of user queries at the customer support center. 
### Our objective is classify the problem into 3 major categories so that we can address them on priority.
############## PROBLEM SOLVING APPROACH
### Step1: take a sample from the queries received ../_data/customer_intent_raw.csv. Ask LLM to find 3 top categories for you. (do multiple run to find the most frequent occuring onces)
###
### Step2: take another sample of data, and label them MANUALLY using the 3 categories you found, label other where there is no match.
###
### Step3: Use the labelled sample data to determine which prompt technique you should use to categories the query. 
### Basically, you are going to evaluate prompt techniques.

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
from utility.llm import LLM

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
llm = LLM.get_client()
chain = llm | parser

class Classification:

    @staticmethod
    def find_top_5_category():
        dataset = pd.read_csv("../_data/customer_intent_raw.csv")

        raw_intent = dataset['description'].str.cat(sep=' || ')

        prompt = f"""Extract the list of problems faced by the user from the following customer queries, separated by " || ". A problem category should be a two word label. List out 10 unique problems
        Then, identify the top 5 most frequent problem encountered by the user.

        Customer queries:
        {raw_intent}"""

        messages = [{'role':'user','content':prompt}]

        response = chain.invoke(messages,
                    config={"temperature": 0, "seed": 49})

        print(response)


user_message_template = """```{description}```"""

few_shot_system_message = """Classify the following product desciption presented in the input into one of the following categories.
    Categories - ['change order', 'track order', 'payment issue']
    Product description will be delimited by triple backticks in the input.
    Answer only 'change order', or 'track order', or 'payment issue'. Nothing Else. Do not explain your answer.
"""
zero_shot_system_message = """Classify the following product desciption presented in the input into one of the following categories.
    Categories - ['change order', 'track order', 'payment issue', 'others']
    Product description will be delimited by triple backticks in the input.
    Answer only 'change order',
    or 'track order', or 'payment issue',or 'others'. Nothing Else. Do not explain your answer.
"""

class PropmtEvaluator:
    _client = None  # Static variable for the LLM client

    @staticmethod
    def __initialize():
        from utility.llm import LLM
        client = LLM.get_client()
        PropmtEvaluator._client = client

    @staticmethod
    def get_llm():
        if PropmtEvaluator._client == None:
            PropmtEvaluator.__initialize()

        return PropmtEvaluator._client
    
    @staticmethod
    def get_chain():
        from langchain_core.output_parsers import StrOutputParser
        parser = StrOutputParser()

        if PropmtEvaluator._client == None:
            PropmtEvaluator.__initialize()

        chain = PropmtEvaluator._client | parser 
        return chain

    @staticmethod
    def __evaluate_prompt(prompt, gold_examples, user_message_template,samples_to_output = 5):

        """
        Return the accuracy score for predictions on gold examples.
        For each example, we make a prediction using the prompt. Gold labels and
        model predictions are aggregated into lists and compared to compute the
        accuracy.

        Args:
            prompt (List): list of messages in the Open AI prompt format
            gold_examples (str): JSON string with list of gold examples
            user_message_template (str): string with a placeholder for description.
            samples_to_output (int): number of sample predictions and ground truths to print

        Output:
            accuracy (float): accuracy score computed by comparing model predictions
                                    with ground truth
        """
        count = 0
        model_predictions, ground_truths = [], []

        # Iterating through all the gold examples and constructing the messages dictionary using the text from example

        for example in json.loads(gold_examples):
            
            gold_input = example['description']
            user_input = [
                {
                    'role':'user',
                    'content': user_message_template.format(description=gold_input)
                }
            ]

            try:
                prediction=PropmtEvaluator.get_chain().invoke(prompt+user_input,
                    config={"temperature": 0, "max_tokens": 4})
                
                while count < samples_to_output:
                    count += 1
                    print(f"{count}-Original label: {example['task']} - Predicated label: {prediction}")

                model_predictions.append(prediction.strip().lower()) # <- removes extraneous white space and lowercases output
                ground_truths.append(example['task'].strip().lower())

            except Exception as e:
                print(e)
                continue

        # Find the accuracy of each category.
        df = pd.DataFrame({
        'Predictions': model_predictions,
        'Ground Truth': ground_truths
        })
        labels = df['Ground Truth'].unique()

        # Create a new DataFrame for the accuracies
        accuracy_df = pd.DataFrame(columns=labels)

        for label in labels:
            # Filter rows where Ground Truth is the current label
            subset = df[df['Ground Truth'] == label]

            # Calculate accuracy for the current label
            accuracy = accuracy_score(subset['Ground Truth'], subset['Predictions'])

            # Add accuracy to the DataFrame
            accuracy_df.loc[0, label] = accuracy

        print("\n\n", accuracy_df)
        print("====================================================")

        accuracy = accuracy_score(ground_truths, model_predictions)

        return accuracy

    @staticmethod
    def __create_examples(examples_df, n=4):

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

        task1 = (examples_df.task == 'change order')
        task2 = (examples_df.task == 'track order')
        task3 = (examples_df.task == 'payment issue')
        task4 = (examples_df.task == 'others')

        t1_examples = examples_df.loc[task1, :].sample(n)
        t2_examples = examples_df.loc[task2, :].sample(n)
        t3_examples = examples_df.loc[task3, :].sample(n)
        t4_examples = examples_df.loc[task4, :].sample(n)
        examples = pd.concat([t1_examples,t2_examples,t3_examples,t4_examples])
        # sampling without replacement is equivalent to random shuffling
        randomized_examples = examples.sample(4*n, replace=False)

        return randomized_examples.to_json(orient='records')

    @staticmethod
    def __create_prompt(system_message, examples, user_message_template):

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

            few_shot_prompt.append(
                {
                    'role': 'user',
                    'content': user_message_template.format(
                        description=example['description']
                    )
                }
            )

            few_shot_prompt.append(
                {'role': 'assistant', 'content': f"{example['task']}"}
            )

        return few_shot_prompt

    @staticmethod
    def evaluate_zero_shot_promt(gold_examples):
        zero_shot_prompt = [{'role':'system', 'content': zero_shot_system_message}]

        token_size = LLM.num_tokens_from_messages(zero_shot_prompt)

        accuracy = PropmtEvaluator.__evaluate_prompt(zero_shot_prompt, gold_examples, user_message_template)

        return token_size, accuracy
    
    @staticmethod
    def evaluate_few_shot_promt(examples_df, gold_examples):

        few_shot_examples = PropmtEvaluator.__create_examples(examples_df, 2)

        few_shot_prompt = PropmtEvaluator.__create_prompt(few_shot_system_message,few_shot_examples,user_message_template)

        token_size = LLM.num_tokens_from_messages(few_shot_prompt)

        accuracy = PropmtEvaluator.__evaluate_prompt(few_shot_prompt, gold_examples, user_message_template)

        return token_size, accuracy


if __name__ == "__main__":
    ##Step1: Find the top 3 category
    #Classification.find_top_5_category()

    ##Step2: Use the top 3 category to sample a set of data in "customer_intent_labeled.csv".

    ##Step3: Now let's evaluate which prompt gives us the accuracy in labeling them automatically. 
    #############################################################################################

    ## Prepare data for example and testing
    dataset_df = pd.read_csv("../_data/customer_intent_labeled.csv")
    examples_df, gold_examples_df = train_test_split(
        dataset_df, #<- the full dataset
        test_size=0.6, #<- 80% random sample selected for gold examples
        random_state=42, #<- ensures that the splits are the same for every session
    )


    num_eval_runs = 1
    few_shot_performance = []
    ## Perform evaluation multiple times with random set of data
    for _ in tqdm(range(num_eval_runs)):

        # gold_examples = json.loads((gold_examples_df.sample(100, random_state=42).to_json(orient='records')))
        gold_examples = (gold_examples_df.sample(100, random_state=42).to_json(orient='records'))
        # print(gold_examples)

        token_size, few_shot_accuracy = PropmtEvaluator.evaluate_few_shot_promt(examples_df, gold_examples)

        few_shot_performance.append(few_shot_accuracy)

