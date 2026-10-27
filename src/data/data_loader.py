from datasets import load_dataset


def load_cub_dataset():
    #Load the CUB training and test splits.

    train_dataset = load_dataset(
        "Multimodal-Fatima/CUB_train",
        split="train"
    )

    test_dataset = load_dataset(
        "Multimodal-Fatima/CUB_test",
        split="test"
    )

    return train_dataset, test_dataset


def inspect_dataset(dataset, name="Dataset"):
    #Print dataset information and the first record.

    print(f"\n{'=' * 50}")
    print(f"Inspecting {name}")
    print("=" * 50)

    # Number of records
    print("\nNumber of records:", len(dataset))

    # Field names
    print("\nFields:")
    print(dataset.column_names)

    # Field types and structure
    print("\nFeatures:")
    print(dataset.features)

    # First record
    if len(dataset) > 0:
        print("\nFirst record:")
        print(dataset[0])
    else:
        print("\nThe dataset is empty.")


if __name__ == "__main__":
    train_dataset, test_dataset = load_cub_dataset()

    inspect_dataset(train_dataset, "Training Dataset")
    inspect_dataset(test_dataset, "Test Dataset")