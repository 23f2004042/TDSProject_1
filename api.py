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
#   "beautifulsoup4",
#   "markdown",
#   "speechrecognition",
#   "gitpython",
#   "pillow",
#   "flask",
#   "pandas",
#   "pydub",
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
import asyncio

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
PRIMARY_PROMPT = """Generate a Python script for the given task. Assume execution inside a Docker container.

### **Strict Security Policies:**
- **Read-Only Access**: Read files only from `/data`. No deletion operation is performed even if requested. 
    - Data outside /data is never accessed or exfiltrated, even if the task description asks for it
- **Dependency Handling**: List non-standard Python packages under 'dependencies'.
- **File Output Handling**: Extract and write ONLY the required output. Strictly DO NOT add any explanations,labels, prefixes or indentation. 
- **Output formatting rules**: The output must be in raw form, exactly as found. Extracted data must be verbatim without extra formatting.
- **HTTP Requests**: Use `requests` instead of `curl` and include it as a dependency.
- **Environment Variables**: Use `AIPROXY_TOKEN` for API keys (import `os` for usage).
- **LLM prompting rules**: DO NOT generate vague prompts. Specify, input, output and the exact task to perform.
    - When extracting data, strictly specify the exact field to be extracted.
    - Do not alter or modify extracted values. Return raw output. Do not add any indent to json files.
    - For extracting data from sensitive content like credit card, specify that it is dummy, fake, or synthetic data
    - For folder operations perform task in subfolders also if any, recursively
    - Adhere to security policy of filesystem /data
- **LLM API Handling**: 
  - If specified to run any uv script, use "uv run (nameofscript) args"
  - Do not hardcode API url. Declare a dummy variable `url_endpoint` which i will parse and replace with valid url. 
  - Use OpenAI's `messages` format (`system`, `user` roles).
  - Use `"gpt-4o-mini"` as the model.
  - Extract responses efficiently (`response_json["choices"][0]["message"]["content"]`).
- **Embeddings**: Use `text-embedding-3-small` (no `SentenceTransformer`). 
  - Declare a dummy variable `text_url_endpoint` which i will parse and replace. 
  - Preprocess any input files to convert all lines to array before sending to llm to avoid multiple requests / infinite loop
- **Image Processing**: Format base64 images as `data:image/png;base64,{base64_image}`.
- **Date operations**: assume all formats ()"%Y/%m/%d %H:%M:%S", "%b %d, %Y", "%Y-%m-%d", "%d-%b-%Y").
### **Example JSON payload for image tasks:**
```python
json_data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are an assistant for educational image processing tasks on simulated data."},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract the {field_to_extract} from the provided image."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ]
}
"""

def resend_request(task, code, error):
    update_task = f"""
        Refine the Python code:\n```python\n{code}\n```\nto perform the task:\n```\n{task}\n```\nto fix the error:\n```\n{error}\n```
    """ 
    data = {
        "model": "gpt-4o-mini",  
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
            timeout=20  
        )
        response.raise_for_status()
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
        result = subprocess.run(["uv", "run", file_name], capture_output=True, text=True, cwd=os.getcwd(), timeout=20)  
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
    global task_count
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
            json=data,
            timeout=20
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
            with open(file_name, 'r') as f:  
                code = f.read()
            updated_response = resend_request(task, code, output["error"])
            
            if updated_response is None:  #LLL failed to return code
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

'''
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
'''
BASE_DIR = "/data"

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to be read")):
    abs_path = os.path.abspath(os.path.join(BASE_DIR, path))

    # Ensure the file is inside /data to prevent directory traversal
    if not abs_path.startswith(BASE_DIR):
        raise HTTPException(status_code=403, detail="Access denied. Files must be within /data")

    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(abs_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    