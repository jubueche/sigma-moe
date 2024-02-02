from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sigma_moe import SigmaMoEForCausalLM


IS_CUDA = torch.cuda.is_available()


def compute_perplexity(model: SigmaMoEForCausalLM, dataloader: DataLoader):
    """
    Compute batched perplexity.
    """
    total_loss = 0
    total_non_empty = 0
    for inputs in tqdm(dataloader):
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to("cuda" if IS_CUDA else "cpu")
        labels = inputs["labels"]
        labels = labels[:, 1:].contiguous()
        labels = labels.to("cuda" if IS_CUDA else "cpu")
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            non_empty_indices = ~(labels == -100).all(1)
            logits = outputs.logits[..., :-1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(
                input=logits[non_empty_indices].transpose(1, 2),
                target=labels[non_empty_indices],
                reduction="none",
            ).sum(1) / (~(labels[non_empty_indices] == -100)).sum(1)
            total_loss += loss.sum()
            total_non_empty += non_empty_indices.sum()
    mean_loss = total_loss / total_non_empty
    return mean_loss.exp()


def main():
    # load the model from the hub
    model = SigmaMoEForCausalLM.from_pretrained("ibm-aimc/sigma-moe-small")
    model = model.eval()
    if IS_CUDA:
        model.cuda()

    # load the tokenizer (sentencepiece)
    tokenizer = AutoTokenizer.from_pretrained("ibm-aimc/sigma-moe-small")

    # load the dataset. Important: This is already tokenized!
    dataset = datasets.load_dataset("ibm-aimc/sentencepiece-wikitext-103")
    
    # create data loader from it
    dataloader = DataLoader(
        dataset=dataset["test"]["input_ids"],
        batch_size=64,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # compute perplexity
    ppl = compute_perplexity(model, dataloader)
    print(f"Perplexity is {ppl:.2f}")


if __name__ == "__main__":
    main()
