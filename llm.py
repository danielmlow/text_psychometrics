import json
import requests
import replicate


def process_api_output(output_str):
  """
  Processes the API output string and returns a dictionary.

  Args:
    output_str: The string output from the API call.

  Returns:
    A dictionary containing the extracted data.
  
  Example:
    
    output_str1 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It wont get better"]] }'
    output_str2 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It wont get better"]] }Explanation: - The text clearly expresses a "desire to escape" with the phrase "I want out", which suggests a strong desire to leave the current situation.- The text also clearly expresses "loneliness" with the phrase "No one cares about me", which indicates feelings of isolation and disconnection.- The text may suggest "suicidal ideation" with the phrases "I want out" and "It wont get better", but it\'s not explicitly stated, hence the lower score.'
    output_str3 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It wont get better"]] }Some additional information here.'

    print(process_api_output(output_str1))
    print(process_api_output(output_str2))
    print(process_api_output(output_str3))
  """
  data = {} 
  start_index = output_str.find('{') 
  end_index = output_str.rfind('}') + 1

  try:
    # Attempt to directly load the JSON string
    data = json.loads(output_str)
  except json.JSONDecodeError:
    # If JSON decoding fails, try to extract the JSON part
    if start_index != -1 and end_index != -1:
      json_part = output_str[start_index:end_index]
      data = json.loads(json_part)
    else:
      raise ValueError("Invalid API output format.")

  # Extract the additional note if it exists
  if start_index != 0 or end_index != len(output_str):
    data['Additional note'] = output_str[end_index:].strip()

  return data



def openrouter_request(prompt, OPENROUTER_API_KEY, model = 'meta-llama/llama-3.1-405b-instruct:free',
                       temperature = 0, safety_settings=None):
  """
  free models: 20 requests per minute and 200 requests per day. See https://openrouter.ai/docs/limits
  """
  provider = model.split('/')[0]

  response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"

        # "provider": json.dumps({
        #   "order": ["openai", "azure"]
        #   }),
      },
      
      data=json.dumps({
        "model": model, # Optional
        "temperature": temperature,
        "response_format": { "type": "json_object" },
        "messages": [
          { "role": "user", "content": prompt}
        ],
        "safety_settings": safety_settings[provider]

      })
      )
  
  try:
    metadata = json.loads(response.text.strip())
    final_result = metadata['choices'][0]['message']['content']
    final_result = dict(eval(final_result))# cleaning: remove underscores from keys
    
    return final_result, metadata

  except:
    try:
        metadata = json.loads(response.text.strip())
        final_result = process_api_output(final_result)
        return final_result, metadata
    
    except Exception as e:
        metadata = response.text
        print('Error:', e)
        print('Could not parse the response, perhaps because the model did not follow the instructions well. gpt-4o,gpt-4o-mini, gemini 1.5, claude 3.5  works well.')
        print("Returning full response with metadata. This is what I'm trying to parse with eval() in content:")
        print(metadata)
        return None, metadata
  


def replicate_request(prompt, REPLICATE_API_TOKEN, model="google-deepmind/gemma-2b", version=None,
                      temperature=0.1,
                      max_new_tokens = 128, top_k=10, top_p=0.10, 
                      # min_new_tokens=-1, repetition_penalty=1.15
                      ):
    """
    Makes a request to the Replicate API using a specified model.
    Returns the full generated output and the metadata (iterator object).
    
    Example:
    prompt = "Write me a poem about Machine Learning."
    REPLICATE_API_TOKEN = "your_api_token_here"

    result, metadata = replicate_request(prompt, REPLICATE_API_TOKEN)
    print(result)
    """
    try:
      client = replicate.Client(api_token=REPLICATE_API_TOKEN)

      output = client.run(
          model, 
          input={
              "prompt": prompt,
              "temperature": temperature,
              "max_new_tokens": max_new_tokens, #good to limit processing time
              # "top_k": top_k,
              #     "top_p": top_p,
              #     "max_new_tokens": max_new_tokens,
              #     "min_new_tokens": min_new_tokens,
              #     "repetition_penalty": repetition_penalty,
          }
      )

      
      final_result = ''.join(output)



      return final_result, None

    except Exception as e:
        print("Error:", e)
        return None, None
