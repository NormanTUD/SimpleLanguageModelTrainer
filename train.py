import argparse

# Create an argument parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--learning-rate', type=float, required=True, default=5e-5,
                    help='the learning rate for the model')
parser.add_argument('--num-epochs', type=int, required=True, default=5,
                    help='the number of epochs to train the model for')
parser.add_argument('--test-string', type=str, required=True,
                    help='the string to test the model on')
parser.add_argument('--filename', type=str, required=True,
                    help='the name of the file to save the model to')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size to use for training (default: 32)')
parser.add_argument('--max-length', type=int, default=50,
                    help='the batch size to use for training (default: 50)')

# Parse the arguments
args = parser.parse_args()

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

def msg (m):
    print(bcolors.WARNING + str(m) + bcolors.ENDC)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

device = torch.device("cpu")
if torch.cuda.is_available():
    msg("Using CUDA")
    device = torch.device("cuda")
else:
    msg("Using CPU")

# Access the values of the arguments
learning_rate = args.learning_rate
num_epochs = args.num_epochs
test_string = args.test_string
filename = args.filename
batch_size = args.batch_size
max_length = args.max_length


# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
if os.path.exists("best.pt"):
    msg("Loaded from best.pt")
    model = AutoModelWithLMHead.from_pretrained("best.pt")
else:
    msg("Loading default dbmdz/german-gpt2")
    model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

def read_file(filename, chunk_size=1024):
    #if chunk_size == 0:
    #    with open(filename, 'r') as f:
    #        return f.read()
    try:
        with open(filename, "r") as f:
            chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                tokenized_chunk = tokenizer.encode(chunk, add_special_tokens=False, return_tensors='pt')
                chunks.append(tokenized_chunk)
            tokenized_data = torch.cat(chunks, dim=1)
    except Exception as e:
        msg(f"Error: {e}")
        tokenized_data = None
    return tokenized_data

msg("Reading file")
data = read_file(filename, 4096)
msg("File read")

last_generated_text = ""

if data is not None:
    # Set the model to training mode
    msg("data is not None")
    msg("Moving model to device")
    model.to(device)
    msg("Switch to training mode")
    model.train()

    msg("Initialize the optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    best_loss = float("inf")

    # Fine-tune the model
    train_loss_history = []

    if num_epochs:
        for epoch in range(num_epochs):
            for i in range(0, data.size(1), batch_size):
                # Forward pass
                inputs = data[:, i:i+1024].to(device)
                outputs = model(inputs, labels=inputs)
                loss = outputs[0].to(device)
                train_loss_history.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                scheduler.step()

                if loss < best_loss:
                    best_loss = loss
                    model.save_pretrained("best.pt")

                if i % 10 == 0:
                    test_tokenized = tokenizer.encode(test_string, return_tensors='pt')
                    generated_text = model.generate(test_tokenized.to(device), max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
                    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

                msg("Epoch %d, batch: %d, loss: %0.15f" % (epoch, i, loss))
                if generated_text != last_generated_text:
                    msg(generated_text)
                    last_generated_text = generated_text

    msg("Set the model to evaluation mode")
    model.eval()

    msg("Generate text")
    test_tokenized = tokenizer.encode(test_string, return_tensors='pt')
    generated_text = model.generate(test_tokenized.to(device), max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(generated_text)
