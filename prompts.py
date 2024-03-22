prompts ={ 
	'google/gemma_2b_it': 
"""
You are a conversation classification assistant. Classify the following {context} conversation:

Here is the {context} conversation (ends with ```):
```
{document}
```

Assign a probability to this conversation for following labels and return using this JSON format (do not provide additional notes, explanations, or warnings). Provide your best guess, only return JSON:

JSON:
{{'texter mentions something somewhat related to {construct}': <your_probability>, 'texter does not mention anything related to {construct}': <your_probability>}}


JSON:
"""
}