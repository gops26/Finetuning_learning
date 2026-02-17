# Finetuning a Openai Model

finetuning a Openai Model is not as complex as it sounds.

It involves simpler workflow to finetune a LLM with a jsonl dataset.

## Fetch dataset

The primary step in finetuning is selecting a appropriate dataset to undergo finetuning. the data should be stored in a jsonl format containing conversation formatted messages.

Each line in the jsonl file should follow this structure:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

You need a minimum of 10 examples, but 50-100+ examples are recommended for meaningful results.

## Validate the dataset

Before uploading, validate your dataset format using the OpenAI cookbook validation script or manually check that:

- Every line is valid JSON
- Each entry has a `messages` key with a list of messages
- Each message has `role` and `content` fields
- Roles follow the order: `system` (optional) -> `user` -> `assistant`

## Upload the dataset

Upload your training file using the OpenAI API:

```python
from openai import OpenAI

client = OpenAI()

training_file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

print(training_file.id)  # file-abc123
```

Optionally, upload a validation file the same way to track overfitting during training.

## Create a finetuning job

Once the file is uploaded, create a finetuning job:

```python
finetune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",  # base model to finetune
    hyperparameters={
        "n_epochs": 3,               # number of training epochs
        "batch_size": "auto",        # let OpenAI decide
        "learning_rate_multiplier": "auto"
    },
    suffix="my-custom-model"         # optional custom name suffix
)

print(finetune_job.id)  # ftjob-abc123
```

Supported base models for finetuning include `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`, and `gpt-3.5-turbo`.

## Monitor the finetuning job

Track the progress of your finetuning job:

```python
# Check job status
job = client.fine_tuning.jobs.retrieve(finetune_job.id)
print(job.status)  # validating_files, running, succeeded, failed

# List events/logs
events = client.fine_tuning.jobs.list_events(finetune_job.id, limit=10)
for event in events.data:
    print(event.message)
```

Training typically takes minutes to hours depending on dataset size and number of epochs.

## Use the finetuned model

Once the job status is `succeeded`, retrieve the finetuned model name and use it like any other OpenAI model:

```python
# Get the finetuned model name
job = client.fine_tuning.jobs.retrieve(finetune_job.id)
finetuned_model = job.fine_tuned_model  # ft:gpt-4o-mini-2024-07-18:org::abc123

# Use it for completions
response = client.chat.completions.create(
    model=finetuned_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Tips

- Start with a small dataset and iterate. Quality matters more than quantity.
- Keep system messages consistent across your training examples.
- Monitor training and validation loss to detect overfitting.
- Use the `suffix` parameter to keep your finetuned models organized.
- Finetuning costs are based on the number of tokens in your training data multiplied by the number of epochs. 