# /// script
# dependencies = [
#   "fastapi",
#   "httpx",
#   "requests",
#   "scikit-learn",
#   "python-dotenv",
#   "uvicorn",
#   "requests<3",
#   "duckdb",
#   "numpy",
#   "python-dateutil",
#   "docstring-parser",
#   "pydantic",
#   "python-dotenv",
#   "beautifulsoup4",
#   "markdown",
#   "docstring-parser",
# ]
# ///

import os
from os.path import join
import subprocess
import requests
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import sys
import pkgutil
import logging
import re

app = FastAPI()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')  

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
task_count = 0
OPEN_AI_TOKEN = os.getenv("AIPROXY_TOKEN")
URL_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
EMB_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
RESPONSE_FORMAT = {  
    "type": "json_schema",
    "json_schema": {
        "name": "task_agent",
        "schema": {  
            "type":"object",
            "required": ["python_dependencies", "python_code"],
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "Python code with proper indendation to perform the task with required imports"
                },
                "python_dependencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module": {
                                "type": "string",
                                "description": "Name of the Python module"
                            }
                        },
                        "required": ["module"],
                        "additionalProperties": False
                    }
                },
                "url_endpoint": {
                    "type": "string",
                    "description": "The API endpoint to be used for LLM requests."
                },
                "text_url_endpoint": {
                    "type": "string",
                    "description": "The API endpoint to be used for embeddings LLM requests."
                },
            }
        }
    }
}

PRIMARY_PROMPT = """Generate a Python script for the given task. Assume that code you generate will be executed inside a Docker container.
- If you need to run any uv script then use "uv run (nameofscript) args"
- If the script requires external Python packages (i.e., not part of the std lib), 
list them under 'dependencies'. 
- For date operations assume all formats ()"%Y/%m/%d %H:%M:%S", "%b %d, %Y", "%Y-%m-%d", "%d-%b-%Y").
- If asked to write output to a file, write only output, do not add indent or pretty print. 
- When writing filename to output file, add relative or absolute path as per asked in task. 
- Use requests library instead of curl and include it as a dependency.
- Do not hardcode LLM endpoint URLs, define a dummy variable `url_endpoint`, which i will parse and replace, do not use any environment variable.
- Use environment variable `AIPROXY_TOKEN` for API keys, and import os package when using environment var. 
- Use the OpenAI API format with a `messages` list.
- Ensure the request uses the `"role": "system"` and `"role": "user"` structure.
- The `"user"` content should contain both `"type": "text"` with task instructions and `"type": "image_url"` to pass base64 url as
    `f"data:image/png;base64,`{base64_image`}"`.
- Construct the JSON request with the model `"gpt-4o-mini"`.
- Request ONLY the required information in LLM query, no other fields. (eg response_json["choices"][0]["message"]["content"])
- For sensitive information, try to send a task description such that LLM responds with required data as the query is on a simulated image
- Use text-embedding-3-small for text similarity check and define only one dummy variable 'text_url_endpoint', do not use SentenceTransformer
- To process embeddings read input file and convert all lines to array before sending to llm to avoid multiple requests

Example JSON payload for image:
```python
json_data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "You are an image processing assistant for educational exercises. Your task is to extract requested info. This is a simulated dataset with no real personal information."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the {field_to_extract} from the provided image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
}
"""

def resend_request(task, code, error):
    update_task = f"""
        Refine the Python code:\n```python\n{code}\n```\nto perform the task:\n```\n{task}\n```\nto fix the error:\n```\n{error}\n```
    """  # Use f-string and clearer formatting
    data = {
        "model": "gpt-4o-mini",  # Or another suitable model
        "messages": [
            {"role": "user", "content": update_task},
            {"role": "system", "content": PRIMARY_PROMPT}
        ],
        "response_format": RESPONSE_FORMAT
    }
    #logging.debug("data", data)
    try:
        response = requests.post(
            URL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {OPEN_AI_TOKEN}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=10  # Add a timeout to prevent indefinite hanging
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with LLM: {e}")
        return None  # Return None to indicate failure


"""Remove built-in modules from the dependencies list."""
def filter_builtin_modules_1(dependencies):
    print("before filter",dependencies )
    std_lib_modules = {module.name for module in pkgutil.iter_modules()}
    filtered_dependencies = []
    includemod = ['requests', 'numpy']
    for dep in dependencies:
        if ((dep["module"] in ['requests', 'numpy'])  or (dep["module"] not in std_lib_modules)):
            filtered_dependencies.append(dep)
    return filtered_dependencies

def replace_url_endpoint(filepath, new_url, new_text_url):
    try:
        with open(filepath, 'r') as f:
            file_content = f.read()
        
        # Regular expressions to match both patterns
        url_pattern = r"^url_endpoint\s*=\s*['\"]?[^'\"]*['\"]?"
        text_url_pattern = r"^text_url_endpoint\s*=\s*['\"]?[^'\"]*['\"]?"

        url_replacement = f"url_endpoint = '{new_url}'"
        text_url_replacement = f"text_url_endpoint = '{new_text_url}'" 

        new_file_content = re.sub(url_pattern, url_replacement, file_content, flags=re.MULTILINE)
        new_file_content = re.sub(text_url_pattern, text_url_replacement, new_file_content, flags=re.MULTILINE)

        if new_file_content == file_content: 
            print(f": 'url_endpoint =' not found in {filepath}. No replacement done.")
            return False
        with open(filepath, 'w') as f:
            f.write(new_file_content)
        print(f"URL endpoint in {filepath} updated to {new_url}")
        return True
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def llm_code_executer(python_dependencies, python_code):
    filtered_dependencies = filter_builtin_modules_1(python_dependencies)
    print("After filter", filtered_dependencies)
    inline_metadata_script = f"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
{''.join(f"# \"{dependency['module']}\",\n" for dependency in filtered_dependencies)}# ] 
# ///
 
"""
    global task_count
    task_count += 1
    try:
        file_name = f"llm_code_task_{task_count}.py"
        with open(file_name, "w") as f:
            f.write(inline_metadata_script)
            f.write(python_code)

        if replace_url_endpoint(file_name, URL_ENDPOINT, EMB_ENDPOINT):
            print("URL replacement successful.")
        else:
            print("URL not replaced")

        with open(file_name, 'r') as f:  
            code = f.read()
        logging.debug("python code after append")
        logging.debug(code)

        # Use subprocess.run with more robust error handling
        result = subprocess.run(["uv", "run", file_name], capture_output=True, text=True, cwd=os.getcwd(), timeout=30)  # Add timeout
        logging.debug(result)
        std_err = result.stderr
        std_out = result.stdout
        exit_code = result.returncode

        if exit_code == 0:
            print("returning success")
            return {"output": "Task execution success"} 
        else:
            logging.error(f"Error executing code (exit code {exit_code}):\n{std_err}")
            return {"error": std_err}  # Return the error message
        
    except subprocess.TimeoutExpired:
        logging.error("Code execution timed out.")
        return {"error": "Code execution timed out."}
    except Exception as e:
        logging.error(f"Exception occurred during code execution: {e}")
        return {"error": str(e)}

@app.post("/run", status_code=status.HTTP_200_OK)
async def task_agent(task: str = Query(..., description="Task description in plain English")):
    data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": task},
                {"role": "system", "content": PRIMARY_PROMPT}
            ],
            "response_format": RESPONSE_FORMAT
    }
    #logging.debug(data)
    response = requests.post(
            URL_ENDPOINT,
            headers={
                "Authorization": f"Bearer {OPEN_AI_TOKEN}",
                "Content-Type": "application/json",
            },
            json=data
    )

    response.raise_for_status()
    r = response.json()
    
    try:
        llm_response_content = json.loads(r['choices'][0]['message']['content'])
        python_dependencies = llm_response_content['python_dependencies']
        python_code = llm_response_content['python_code']
        #logging.debug(f"Python dependencies: {python_dependencies}, python code: {python_code}")

    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error parsing LLM response: {e}, Response content: {r}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Invalid LLM response format"})

    output = llm_code_executer(python_dependencies, python_code)
    logging.debug("output",output)

    retries = 0
    max_retries = 2
    while retries < max_retries:
        if "output" in output:
            return JSONResponse(status_code=status.HTTP_200_OK, content={"output": output["output"]})
        elif "error" in output:
            file_name = f"llm_code_task_{task_count}.py"
            with open(file_name, 'r') as f:  # Read the generated code
                code = f.read()
            updated_response = resend_request(task, code, output["error"])
            if updated_response is None:  # Handle LLM communication failure
                return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Failed to communicate with LLM for retry."})
            try:
                r = updated_response.json()
                llm_response_content = json.loads(r['choices'][0]['message']['content'])
                python_dependencies = llm_response_content['python_dependencies']
                python_code = llm_response_content['python_code']
                output = llm_code_executer(python_dependencies, python_code)
                logging.debug("output",output)
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Error parsing updated LLM response: {e}, Response content: {r}")
                return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Invalid updated LLM response format"})
            retries += 1
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Unexpected output from code execution."})
    if ("output" in output):
        return JSONResponse(status_code=status.HTTP_200_OK, content={"output": output["output"]})
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"Task failed after {max_retries} retries."})

@app.get("/")
def home():
    return "Welcome to Task Agent"

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to be read")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)