import os
import pandas as pd
import argparse
from dotenv import load_dotenv
import dspy
from typing import Literal
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFewShot, MIPROv2

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Set model name
model = "openrouter/meta-llama/llama-3-70b-instruct"

# Configure DSPy LLM using OpenRouter
lm = dspy.LM(model, api_key=api_key)

dspy.configure(lm=lm)


def read_dataset(file_path: str, debug: bool):
    """Reads the dataset from a TSV file and selects the first 10 rows if debug is enabled."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} does not exist.")

    df = pd.read_csv(file_path, sep='\t', header=0, quoting=3)
    if debug:
        df = df.head(10)
    print(f"Loaded dataset from {file_path} preview:")
    print(df)
    return df


def create_examples(df: pd.DataFrame):
    """Generates DSPY examples from the dataset and stores them in a list."""
    if df.empty:
        raise ValueError("The dataset is empty; cannot create examples.")

    dataset = []
    for _, row in df.iterrows():
        sentence = row["text"]
        label = str(row["label"])
        example = dspy.Example(sentence=sentence, label=label).with_inputs("sentence", "label")
        dataset.append(example)
    return dataset


class Sentiment(dspy.Signature):
    """Binary sentiment classification."""
    sentence: str = dspy.InputField()
    label: Literal[0, 1] = dspy.OutputField()


def classify_sentences(dataset, optimized_program=None, evaluation_phase="Initial"):
    """Runs DSPY signature tasks on the dataset and calculates success rate."""
    classify = optimized_program if optimized_program else dspy.Predict(Sentiment)
    correct_predictions = 0

    for example in dataset:
        response = classify(sentence=example.sentence)
        predicted_label = str(response.label)
        true_label = str(example.label)

        print(f"Sentence: {example.sentence}")
        print(f"Predicted label: {predicted_label}, True label  : {true_label}")

        if predicted_label == true_label:
            correct_predictions += 1

    success_rate = correct_predictions / len(dataset) * 100 if dataset else 0
    print("")
    print(f"{evaluation_phase} success rate: {success_rate:.2f}%\n")
    return success_rate


def optimize_program(trainset):
    """Optimizes the DSPy program using BootstrapFewShotWithRandomSearch."""
    if not trainset:
        raise ValueError("The training set is empty; cannot run optimization.")

    config = {
        "max_bootstrapped_demos": 4,
        "auto": "light",
        #"max_labeled_demos": 4,
        #"max_rounds": 10
        ##"num_candidate_programs": 10,
        ##"num_threads": 4
    }

    # Define a metric function for optimization
    metric = (lambda x, y, trace=None: x.label == y.label)

    # Set up the optimizer
    teleprompter = MIPROv2(metric=metric, **config)
    signature_task = dspy.Predict(Sentiment)
    optimized_program = teleprompter.compile(signature_task, trainset=trainset)

    print("Optimization completed.")
    return optimized_program


def save_optimized_program(optimized_program, save_path: str):
    """Saves the optimized DSPy program."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimized_program.save(save_path)
    print(f"Optimized program saved at: {save_path}")


def compute_total_cost():
    """Computes and prints the total cost of program execution."""
    if not lm.history:
        print("No cost history available.")
        return

    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    print(f"Total cost of running the script: ${cost:.2f}")


def main(args):
    train_df = read_dataset('train.tsv', args.debug)

    # Create DSPy examples for training and testing (only on the training set)
    trainset = create_examples(train_df)

    # Initial evaluation before optimization
    print("Initial evaluation on the training set...")
    classify_sentences(trainset, evaluation_phase="Initial")

    # Optimization logic
    if args.optimize:
        print("Running optimization...")
        optimized_program = optimize_program(trainset)

        # Save the optimized program and compute total cost
        save_optimized_program(optimized_program, "output/optimized_program.json")

        # Post-optimization evaluation
        print("Post-optimization evaluation on the training set...")
        classify_sentences(trainset, optimized_program=optimized_program, evaluation_phase="Post-Optimization")

    compute_total_cost()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPy tasks and compute success rate.")
    parser.add_argument("--optimize", action="store_true", help="Enable program optimization")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (use only the first 10 rows of the dataset)")
    args = parser.parse_args()

    main(args)
