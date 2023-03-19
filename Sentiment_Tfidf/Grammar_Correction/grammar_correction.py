import openai

openai.api_key = "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"

def grammar_correction(text):
    """
    Correct the grammar of a sentence using GPT-3

    Args:
        text (str): sentence to correct
    
    Returns:
        corrected_text (str): corrected sentence
    """

    ## Initialize the chat completion and get a response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"Can you correct the grammar of the following sentence? '{text}'"}
        ]
    )
    
    ## Get the corrected text
    corrected_text = dict(completion.choices[0].message)["content"].replace("\n", "")
    
    return corrected_text