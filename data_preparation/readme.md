This folder contains scripts used for instruction generation and data preparation in general 

The base prompt used for instruction generation was:
```
"Here is a python dictionary describing some audio from a dataset that describes {dataset_desc}. Come up with ONLY 1 question to these descriptions, and then come up with a nice answer to it. You only need short questions and answers, nothing more, without any notes, and don't mention anything about the dataset and dictionary. Don't put information from dictionary to question, you need to put this information to answer. Write “Question:” before the question, “Answer:” before the answer (that need for parsing). Dictionary: {dictionary}"
```

`Dictionary` contained available metadata for each sample. Depending on set, some metadata wss ommited from dictionary if that data tend to break instruction generation, or wasn't useful for instructions in general. 

