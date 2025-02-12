############## PROBLEM STATEMENT
### We are receving product descriptions, and we want to automate a process to auto-level them (Hair Care, Skin Care) using LLM.
### To start with, I will take a sample data and manually label them ""../_data/auto-labelling.csv"".
### Now I will use this sample data to see if my LLM is labeling them as accurately as I have done manually.
### Our objective is find out the suitable LLM, and prompt technique to automate the process.
############## PROBLEM SOLVING APPROACH
### Step1: take a sample from the queries received ../_data/auto-labelling.csv, and split it randomly into example and gold data. 
###
### Step2: Run zero-shot, few-shot, and cot prompt on the gold data and check the accuracy to decide the best prompt technique. (use the example data for few-shot examples)
###
### Step3: You can repeat the step#2 on another LLM to check the accuracy to make your call.
###
### Step4: In the program, I am also performing a comparision between few-shot and cot prompting by executing the test 5 times, and finding the standard deviation in it.

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

user_message_template = """```{product_description}```"""

few_shot_system_message = """
    Classify the following product desciption presented in the input into one of the following categories.
    Categories - ['Hair Care', 'Skin Care']
    Product description will be delimited by triple backticks in the input.
    Answer only 'Hair Care' or 'Skin Care'. Do not explain your answer.
"""
zero_shot_system_message = """
    Classify the following product desciption presented in the input into one of the following categories.
    Categories - ['Hair Care', 'Skin Care']
    Product description will be delimited by triple backticks in the input.
    Answer only 'Hair Care' or 'Skin Care'. Nothing Else. Do not explain your answer.
"""

cot_system_message = """
    Given the following product description, follow these steps to determine the appropriate product label category:

    1. Read the product description carefully, looking for key words and phrases that indicate the product's purpose and usage.

    2. Consider if the description mentions any particular keywords relating to a category.

    3. If the description contains keywords related to multiple categories, determine which category is most strongly emphasized or which usage is primary.

    4. If the description does not contain any clear keywords related to the given categories, consider the overall context and purpose of the product to make an educated guess about the most appropriate category.

    5. Output the determined category label ( 'Hair Care', or 'Skin Care') and nothing else. Do not explain your output.
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
    def __evaluate_prompt(prompt, gold_examples_data, user_message_template,samples_to_output = 1):

        """
        Return the accuracy score for predictions on gold examples.
        For each example, we make a prediction using the prompt. Gold labels and
        model predictions are aggregated into lists and compared to compute the
        accuracy.

        Args:
            prompt (List): list of messages in the Open AI prompt format
            gold_examples_data (str): JSON string with list of gold examples
            user_message_template (str): string with a placeholder for product description
            samples_to_output (int): number of sample predictions and ground truths to print

        Output:
            accuracy (float): Accuracy computed by comparing model predictions
                                    with ground truth
        """

        count =0
        model_predictions, ground_truths = [], []

        for example in json.loads(gold_examples_data):
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

                prediction=PropmtEvaluator.get_chain().invoke(prompt+user_input,
                    config={"temperature": 0.5, "max_tokens": 4})

                # print(prediction) #uncomment to see LLM response or to debug
                model_predictions.append(prediction.strip().lower()) # <- removes extraneous white space and lowercases output
                ground_truths.append(example['Category'].strip().lower())

                if count < samples_to_output:
                    count += 1
                    print(f"{count}-Product Description: {example['Product Description']}-Original label: {example['Category']} - Predicated label: {prediction}")

            except Exception as e:
                print(e)
                continue

            accuracy = accuracy_score(ground_truths, model_predictions)

        return accuracy

    @staticmethod
    def __create_examples(dataset, n=4):

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

    @staticmethod
    def evaluate_zero_shot_promt(gold_examples_data):
        zero_shot_prompt = [{'role':'system', 'content': zero_shot_system_message}]

        token_size = LLM.num_tokens_from_messages(zero_shot_prompt)

        accuracy = PropmtEvaluator.__evaluate_prompt(zero_shot_prompt, gold_examples_data, user_message_template)

        return token_size, accuracy
    
    @staticmethod
    def evaluate_few_shot_promt(examples_df, gold_examples_data):

        few_shot_examples = PropmtEvaluator.__create_examples(examples_df, 2)

        few_shot_prompt = PropmtEvaluator.__create_prompt(few_shot_system_message,few_shot_examples,user_message_template)

        token_size = LLM.num_tokens_from_messages(few_shot_prompt)

        accuracy = PropmtEvaluator.__evaluate_prompt(few_shot_prompt, gold_examples_data, user_message_template)

        return token_size, accuracy


    @staticmethod
    def evaluate_cot_promt(examples_df, gold_examples_data):

        few_shot_examples = PropmtEvaluator.__create_examples(examples_df, 2)

        cot_few_shot_prompt = PropmtEvaluator.__create_prompt(cot_system_message,few_shot_examples,user_message_template)

        token_size = LLM.num_tokens_from_messages(cot_few_shot_prompt)

        accuracy = PropmtEvaluator.__evaluate_prompt(cot_few_shot_prompt, gold_examples_data, user_message_template)

        return token_size, accuracy

    @staticmethod
    def compare_few_shot_vs_cot_promt(examples_df, gold_examples_data):
        num_eval_runs = 5
        few_shot_performance, cot_few_shot_performance = [], []

        for _ in tqdm(range(num_eval_runs)):
            # For each run create a new sample of examples
            examples = PropmtEvaluator.__create_examples(examples_df)

            # Assemble the few shot prompt with these examples
            few_shot_prompt = PropmtEvaluator.__create_prompt(few_shot_system_message, examples, user_message_template)
            cot_few_shot_prompt = PropmtEvaluator.__create_prompt(cot_system_message, examples, user_message_template)

            # Evaluate prompt accuracy on gold examples
            few_shot_accuracy = PropmtEvaluator.__evaluate_prompt(few_shot_prompt, gold_examples_data, user_message_template)
            cot_few_shot_accuracy = PropmtEvaluator.__evaluate_prompt(cot_few_shot_prompt, gold_examples_data, user_message_template)

            few_shot_performance.append(few_shot_accuracy)
            cot_few_shot_performance.append(cot_few_shot_accuracy)

        few_shot_std= np.array(few_shot_performance).mean(), np.array(few_shot_performance).std()
        cot_few_shot_std=np.array(cot_few_shot_performance).mean(), np.array(cot_few_shot_performance).std()

        print(f"Standard deviation in few-shot-prompt performance {few_shot_std}.")
        print(f"Standard deviation in cot-few-shot-prompt performance {cot_few_shot_std}.")


if __name__ == "__main__":

    ## Prepare data for example and testing
    data = pd.read_csv("../_data/auto-labelling.csv")
    examples_df, gold_examples_df = train_test_split(
        data, #<- the full dataset
        test_size=0.8, #<- 80% random sample selected for gold examples
        random_state=42, #<- ensures that the splits are the same for every session
        stratify=data['Category'] #<- ensures equal distribution of labels
    )

    gold_examples_data = (
        gold_examples_df.to_json(orient='records')
    )

    # token_size, accuracy = PropmtEvaluator.evaluate_zero_shot_promt(gold_examples_data)
    # print(f"zero-shot-promt token size is {token_size} and accuracy is {accuracy}")


    # token_size, accuracy = PropmtEvaluator.evaluate_few_shot_promt(examples_df,gold_examples_data)
    # print(f"few-shot-promt token size is {token_size} and accuracy is {accuracy}")


    # token_size, accuracy = PropmtEvaluator.evaluate_cot_promt(examples_df,gold_examples_data)
    # print(f"cot-promt token size is {token_size} and accuracy is {accuracy}")

    PropmtEvaluator.compare_few_shot_vs_cot_promt(examples_df,gold_examples_data)

