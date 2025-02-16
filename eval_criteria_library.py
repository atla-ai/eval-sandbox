EXAMPLE_METRICS = {
"Custom":  {
    "prompt":
    """Evaluate a chat bot response to a user's input based on [INSERT CRITERIA OR SELECT EXAMPLE FROM LIST]

0: The response's answer does not meet all of the evaluation criteria.
1: The response's answer meets all of the evaluation criteria.""",
    },
    "Relevance": {
        "prompt": """Evaluate how well the response fulfill the requirements of the instruction by providing relevant information. This includes responding in accordance with the explicit and implicit purpose of given instruction.

1: The response is completely unrelated to the instruction, or the model entirely misunderstands the instruction.
2: Most of the key points in the response are irrelevant to the instruction, and the response misses major requirements of the instruction.
3: Some major points in the response contain irrelevant information or miss some requirements of the instruction.
4: The response is relevant to the instruction but misses minor requirements of the instruction.
5: The response is perfectly relevant to the instruction, and the model fulfills all of the requirements of the instruction.""",
    },
    "Correctness": {
        "prompt": """Evaluate whether the information provided in the response is correct given the reference response. Ignore differences in punctuation and phrasing between the student answer and true answer. It is okay if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.

0: The response is not factually accurate when compared against the reference response or includes conflicting statements.
1: The response is supported by the reference response and does not contain conflicting statements.""",
    },
    "Helpfulness": {
        "prompt": """Evaluate how helpful the response is to address the user query.

1: The response is not at all useful, failing to address the instruction or provide any valuable information.
2: The response has minimal usefulness, addressing the instruction only superficially or providing mostly irrelevant information.
3: The response is moderately useful, addressing some aspects of the instruction effectively but lacking in others.
4: The response is very useful, effectively addressing most aspects of the instruction and providing valuable information.
5: The response is exceptionally useful, fully addressing the instruction and providing highly valuable information.""",
    },
    "Faithfulness": {
        "prompt": """Evaluate how well the statements in the response are directly supported by the context given in the related passages.

1: The response contains statements that directly contradict the context or are entirely unsupported by it.
2: The response includes some information from the context, but contains significant ungrounded claims or misinterpretations.
3: The response is mostly grounded in the context, with only minor unsupported claims or misinterpretations.
4: The response closely aligns with the context, with only rare and minor deviations.
5: The response is fully grounded in the context, with all statements accurately reflecting the provided information.""",
    },
    "Logical coherence": {
        "prompt": """Evaluate how logically accurate and correct the response is for the instruction given.

1: The logic of the model’s response is completely incoherent.
2: The model’s response contains major logical inconsistencies or errors.
3: The model’s response contains some logical inconsistencies or errors, but they are not significant."
4: The model’s response is logically sound, but it is slightly flawed in some aspect.
5: The model’s response is logically flawless.""",
    },
    "Conciseness": {
        "prompt": """Evaluate how concise the response is presented to the user without any unncecessary information.

1: The response is highly redundant or contains a lot of unnecessary information, requiring a complete rewrite for optimal clarity and efficiency.
2: The response lacks conciseness and needs a substantial rewrite for better optimization.
3: The response is somewhat concise but includes unnecessary information, requiring
some edits for improved optimization.
4: The response is mostly concise but could benefit from minor edits for better optimization.
5: The response is optimally concise and does not contain any unnecessary information, requiring no further optimization.""",
    },
}
