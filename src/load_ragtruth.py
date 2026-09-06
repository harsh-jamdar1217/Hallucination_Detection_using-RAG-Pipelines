import json
from datasets import load_dataset


def load_ragtruth():
    dataset = load_dataset("wandb/RAGTruth-processed")
    return dataset


def get_binary_label(example):
    labels = json.loads(example["hallucination_labels"])
    return 1 if len(labels) > 0 else 0


if __name__ == "__main__":
    dataset = load_ragtruth()
    print(dataset)

    example = dataset["train"][0]
    print("\nType of hallucination_labels:", type(example["hallucination_labels"]))
    print("Value:", example["hallucination_labels"])
    print("Parsed binary label for this example:", get_binary_label(example))

    # Class balance check across the whole train split
    labels = [get_binary_label(ex) for ex in dataset["train"]]
    print(f"\nHallucinated: {sum(labels)} / {len(labels)} ({100*sum(labels)/len(labels):.1f}%)")