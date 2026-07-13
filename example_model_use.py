# ssh into the paderborn hpc, it can't be exposed as an API because the server is locked from paderborn side
# run the below command in a new terminal, put your private key and password, type 'yes'
# Then the model is exposed at port 8000 (keep this port empty)
# $ ssh -i ./id_ed25519 -o ProxyCommand="ssh -i ./id_ed25519 -W %h:%p ankitah@fe.noctua2.pc2.uni-paderborn.de" -L 8000:n2dgx01:8000 ankitah@n2login3
# If the model is off then follow the guide on gitlab main branch readme HPC section
# to load it again on hpc, as it can't be kept running forever as University doesn't allow using hpc forever.


# pip install openai
# use code as below

import openai

# The base_url points to localhost because your SSH tunnel is routing it to the HPC
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="hpc-token"  # vLLM doesn't validate this, any string works
)

def run_refactoring_test(prompt_text):
    try:
        response = client.chat.completions.create(
            model="./models/Qwen2.5-Coder-14B-Instruct",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an automated software architecture refactoring assistant."
                },
                {
                    "role": "user", 
                    "content": prompt_text
                }
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to HPC model: {e}"

# --- Test it right now ---
if __name__ == "__main__":
    sample_code = """
def process_data(items):
    output = []
    for i in range(len(items)):
        if items[i] not in output:
            output.append(items[i])
    return output
"""
    test_prompt = f"Refactor this Python code for better performance and architectural clean lines:\n{sample_code}"
    
    print("Sending request to Noctua 2 cluster...")
    result = run_refactoring_test(test_prompt)
    print("\n--- Refactoring Output ---")
    print(result)