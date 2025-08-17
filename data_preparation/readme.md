This folder contains some materials related to instruction generation and data preparation in general, as well as examples of training data.

Train/validation data is a JSON file that follows the structure in `example.json`. Each sample should contain the keys `paths`, `task`, and `text`. If the `task` field contains a value not present in `cryfish/prompts/train_prompt2.json`, then it should have `"QA"` value and the field `Q` with the user question must be provided.


For instruction generation, we used different models from the Qwen 2.5 family. The LLM was tasked with creating a question–answer pair. We tuned the prompt and dataset description until manual checks of the generated data revealed no hallucinations. Additionally, keyword-based filtering was used to remove questions where metadata leaked into the instructions.\\
The base prompt used for instruction generation was:
```
"Here is a python dictionary describing some audio from a dataset that describes {dataset_desc}. Come up with ONLY 1 question to these descriptions, and then come up with a nice answer to it. You only need short questions and answers, nothing more, without any notes, and don't mention anything about the dataset and dictionary. Don't put information from dictionary to question, you need to put this information to answer. Write “Question:” before the question, “Answer:” before the answer (that need for parsing). Dictionary: {dictionary}"
```


`Dictionary` contained the available metadata for each sample. Depending on the dataset, some metadata was omitted from the dictionary if it tended to break instruction generation or wasn't useful for instructions in general.

