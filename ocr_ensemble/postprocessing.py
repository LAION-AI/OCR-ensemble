import openai

def postprocess(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a masterful text postprocessor."},
            {"role": "user", "content": "I need you to simulate a text postprocessor. When I write postprocess('texttisbetter') you respond with a 'text is better' replacing the input to postprocess with the most plausible text. No explanations, no small talk, no questions, no excuses, only the result. If the input is too hard, just return the input. postprocess('BECOMINGMORE')"},
            {"role": "assistant", "content": "BECOMING MORE"},
            {"role": "user", "content": "postprocess('%s')"%text}
        ]
    )
    return response["choices"][0]["message"]["content"]