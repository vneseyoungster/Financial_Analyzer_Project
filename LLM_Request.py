#!/usr/bin/env python
# Simple LLM Request Handler

import requests
import json
import time
from typing import Dict, Any, List

class LLMRequest:
    """Simple class to handle requests to a local LLM server"""
    
    def __init__(self, base_url: str = "http://192.168.1.119:1234", default_timeout: int = 200):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/v1/chat/completions"
        self.default_timeout = default_timeout
    
    def check_server(self) -> bool:
        """Check if the LLM server is running"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def process_text(self, text: str, max_retries: int = 3, timeout: int = None) -> Dict[str, Any]:
        """
        Send text to LLM for processing with retry mechanism
        
        Args:
            text: Input text to process
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds (overrides default)
            
        Returns:
            dict: LLM response
        """
        if timeout is None:
            timeout = self.default_timeout
            
        system_prompt = """You are a smart financial accountant. You are given a text extracted from a financial document in the Assets section.
You are thinking about how to take out the financial information that is valueable to capture the financial condition of the company. 
.The collected information should be significant for fundamentals analysis. Then you return the information in a markdown format.

Sample Output Format:

### Key Assets Overview (in VND million)

| Asset Category                                      | As at 30/6/2022 | As at 31/12/2021 |
|-----------------------------------------------------|-----------------|------------------|
| **Cash, gold, silver and gemstones**                | 15,097,807      | 18,011,766       |
| **Balances with the State Banks**                   | 28,813,961      | 22,506,711       |
| **Balances with other credit institutions**         | 206,455,463     | 181,036,981      |
| **Loans to other credit institutions**              | 50,081,519      | 48,727,565       |
| **Provision for balances with and loans to others** | (1,000,000)     | 4,000,000        |
| **Trading securities**                              | 3,150,052       | 2,766,098        |
| **Derivatives and other financial assets**          | 303,202         | N/A              |
| **Loans to customers**                              | 1,066,990,245   | 934,774,287      |
| **Provision for loans to customers**                | (33,861,918)    | (25,975,668)     |
| **Investment securities**                           | 191,407,933     | 170,604,700      |
| **Available-for-sale securities**                   | 101,203,452     | 71,122,502       |
| **Held-to-maturity securities**                     | 90,293,045      | 99,657,595       |
| **Provision for investment securities**             | (88,564)        | (175,397)        |
| **Capital contributions and long-term investments** | 2,380,804       | 2,346,176        |
| **Fixed assets**                                    | 8,103,519       | 8,626,043        |
| **Tangible fixed assets**                           | 5,249,947       | 5,552,624        |
| **Intangible fixed assets**                         | 2,853,572       | 3,073,419        |
| **Other assets**                                    | 30,199,661      | 28,969,058       |

- If the information is not available, please return "N/A"
- Don't fabricate or make up any information

"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
            "stream": False  # Ensure we're not using streaming which can cause timeouts
        }
        
        for attempt in range(max_retries):
            try:
                print(f"LLM request attempt {attempt+1}/{max_retries}...")
                
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"]["content"]
                        return {"success": True, "content": content}
                    else:
                        print(f"Unexpected response structure: {result}")
                
                error_msg = f"Request failed with status {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f": {response.text[:200]}"
                
                print(error_msg)
                
                # Only retry if not the last attempt
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    next_timeout = timeout + 30  # Increase timeout for next attempt
                    print(f"Request timed out after {timeout}s. Retrying with {next_timeout}s timeout...")
                    timeout = next_timeout
                    time.sleep(2)
                else:
                    return {"success": False, "error": f"Request timed out after {max_retries} attempts"}
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Maximum retry attempts reached"}


class Financial_Agent:
    """Class to handle financial analysis requests to a local LLM server"""
    
    def __init__(self, base_url: str = "http://192.168.1.119:1234", default_timeout: int = 200):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/v1/chat/completions"
        self.default_timeout = default_timeout
    
    def check_server(self) -> bool:
        """Check if the LLM server is running"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def analyze_financial_data(self, text: str, max_retries: int = 3, timeout: int = None) -> Dict[str, Any]:
        """
        Send financial text to LLM for analysis with retry mechanism
        
        Args:
            text: Input financial text to analyze
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds (overrides default)
            
        Returns:
            dict: LLM response with financial analysis
        """
        if timeout is None:
            timeout = self.default_timeout
            
        system_prompt = """You are a sophisticated financial analyst with expertise in investment analysis, risk assessment, and financial forecasting.
Your task is to analyze the provided financial data. Your main task is to find and extract the income statement within the document for further analysis.

These are the information you need to extract:
- Revenue: known as income of multiple sources, is the total amount of money earned by a company from all business activities before minus the expenses.
- Cost: This represents the cost or the expenses incurred by the company to generate the revenue
- Gross Profit: Profit before taxes
- Operating Expenses: These include costs related to sales, marketing, research and development, and administrative expenses
- Operating Income: Also known as earnings before interest and taxes (EBIT), operating income is gross profit minus operating expenses
- Net Income: This is the bottom line, the profit after all expenses have been deducted from the revenue


- Each information should be format into a key/value pair.
Formatted your response in a json format below:

{
    "Revenue": {
        "value": 100000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    },
    "Cost": {
        "value": 50000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    },
    "Gross Profit": {
        "value": 50000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    },
    "Operating Expenses": {
        "value": 20000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    },
    "Operating Income": {
        "value": 30000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    },
    "Net Income": {
        "value": 20000,
        "from": "2022-01-01",
        "to": "2022-12-31"
    }
}
### GIVE OUT THE FORMULAS USED TO CALCULATE THE VALUES.
### IF YOU CANNOT FIND THE INFORMATION, PLEASE RETURN "N/A" FOR THE KEY/VALUE PAIR.

"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more precise financial analysis
            "max_tokens": 2000,  # Increased token limit for comprehensive reports
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                print(f"Financial analysis request attempt {attempt+1}/{max_retries}...")
                
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"]["content"]
                        return {"success": True, "content": content}
                    else:
                        print(f"Unexpected response structure: {result}")
                
                error_msg = f"Request failed with status {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f": {response.text[:200]}"
                
                print(error_msg)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    next_timeout = timeout + 30
                    print(f"Request timed out after {timeout}s. Retrying with {next_timeout}s timeout...")
                    timeout = next_timeout
                    time.sleep(2)
                else:
                    return {"success": False, "error": f"Request timed out after {max_retries} attempts"}
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Maximum retry attempts reached"}


def main():
    """Main function to demonstrate usage"""
    # Test LLMRequest
    llm = LLMRequest()
    
    if not llm.check_server():
        print("Error: LLM server is not running")
        return
    
    # Example usage of LLMRequest
    print("\n===== Testing LLMRequest =====")
    sample_text = "This is a sample document to analyze."
    result = llm.process_text(sample_text)
    
    if result["success"]:
        print("LLM Response:", result["content"])
    else:
        print("Error:", result["error"])
    
    # Test Financial_Agent
    print("\n===== Testing Financial_Agent =====")
    financial_agent = Financial_Agent()
    
    if not financial_agent.check_server():
        print("Error: LLM server is not running for Financial_Agent")
        return
    
    # Example usage of Financial_Agent
    financial_text = """
    Company XYZ Financial Summary (in millions USD):
    Revenue: $532.4 (2022), $498.2 (2021)
    Net Income: $78.6 (2022), $65.3 (2021)
    Total Assets: $1,245.8 (2022), $1,123.6 (2021)
    Total Liabilities: $523.9 (2022), $567.4 (2021)
    Operating Cash Flow: $112.5 (2022), $98.7 (2021)
    """
    
    result = financial_agent.analyze_financial_data(financial_text)
    
    if result["success"]:
        print("Financial Analysis:", result["content"])
    else:
        print("Error:", result["error"])

if __name__ == "__main__":
    main() 