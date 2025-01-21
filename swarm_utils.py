import re
import json
import httpx
import ollama
  
# Not in use yet
class LogLLMManager:
    """Manages LLM-based log analysis."""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "mistral"
        self._client = None
        self.timeout = 60.0  # Overall timeout for the request

    @property
    def client(self):
        if self._client is None:
            # Increase client timeout settings
            timeout_settings = httpx.Timeout(
                connect=10.0,    # Connection timeout
                read=40.0,      # Read timeout
                write=10.0,     # Write timeout
                pool=10.0       # Pool timeout
            )
            self._client = httpx.Client(timeout=timeout_settings)
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def analyze_logs(self, log_data: str) -> str:
        """Analyze logs using the LLM model with timeout."""
        print("\n=== LLM Log Analysis Started ===")
        try:
            print(f"Sending request to {self.base_url}/api/generate...")
                # print(f"Prompt ----> {prompt}",end="\n\n")

            response = self.client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "analyse follwoing log files",
                        "stream": False
                    }
                )
            response.raise_for_status()
            result = response.json()["analysis"]
            print("Log analysis received successfully")
            return result

        except httpx.ReadTimeout:
            print("ERROR: Read timeout - Ollama is taking too long to respond")
            print("Please check if Ollama is running and the model is loaded")
            return ""
        except httpx.ConnectTimeout:
            print("ERROR: Connection timeout - Cannot connect to Ollama")
            print("Please check if Ollama is running at", self.base_url)
            return ""
        except httpx.ConnectError as e:
            print(f"ERROR: Connection failed - {str(e)}")
            print("Please check if Ollama is running at", self.base_url)
            return ""
        except Exception as e:
            print(f"ERROR: LLM log analysis failed - {str(e)}")
            print(f"Exception type: {type(e)}")
            if hasattr(e, 'response'):
                print(f"Response content: {e.response.content}")
            return ""

def extract_metrics(llm_response, system_state):    
    """    
    Extracts 'nodes' and 'expected_runtime' from the LLM response.    
    
    Parameters:    
        llm_response (str): The full text response from the LLM.    
    
    Returns:    
        tuple: A tuple containing (nodes, expected_runtime) if found, else (None, None).    
    """    
    # Define a regex pattern to capture JSON within ```json``` blocks or plain JSON  
    json_pattern = r"(?:```json\s*({.*?})\s*```|({.*?}))"    
        
    # Search for the JSON block    
    match = re.search(json_pattern, llm_response, re.DOTALL)    
        
    if match:    
        json_str = match.group(1) or match.group(2)  # Capture either group  
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)  # Remove comments  
        try:    
            # Parse the JSON string    
            data = json.loads(json_str)    
                
            # Extract the desired metrics    
            nodes = data.get("node_allocation")    
            expected_runtime_expr = data.get("runtime_estimation_formula")    
              
            # Replace variables in the expression with actual values from system_state  
            # Define the available variables  
            variables = {  
                # "base_time": system_state["job_requirements"]["base_time"],  
                "base_time": system_state["base_time"],
                "min_nodes":system_state["min_nodes"],
                "max_nodes":system_state["max_nodes"],
                "total_nodes": system_state["total_nodes"],
                "available_nodes": system_state["available_nodes"],
                "node_allocation": nodes,
                "allocated_nodes": nodes 
            }  
            # print('PARSED_ variables ----> ',variables)
            # Safely evaluate the expression  
            try:  
                expected_runtime = eval(expected_runtime_expr,{"__builtins__": {}}, variables)  
            except Exception as e:  
                print(f"Error evaluating runtime expression: {e}")  
                expected_runtime = None  
      
            return nodes, expected_runtime    
        except json.JSONDecodeError:    
            print("Error: Invalid JSON format.")    
            return None, None    
    else:    
        print("Error: JSON block not found in the response.")    
        return None, None  





