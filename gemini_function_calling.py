from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from openai import OpenAI
import requests
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import pytz
from geopy.geocoders import Nominatim

def truncate_response(response, max_chars=5000):
    """
    Truncate the response if it exceeds the max character limit.
    """
    response_str = json.dumps(response)
    if len(response_str) > max_chars:
        truncated_str = response_str[:max_chars] + '... (truncated)'
        try:
            truncated_data = json.loads(truncated_str)
        except json.JSONDecodeError:
            truncated_data = {"error": "Response was too large and truncated."}
        return truncated_data
    return response

# Load the .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    ai_response: str
    function_called: str | None = None
    function_args: dict | None = None
    function_response: dict | str | float | None = None

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define API Key authentication
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)
FUNCTION_CALLING_API_KEY = os.getenv("FUNCTION_CALLING_API_KEY")

# API key verification function
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != FUNCTION_CALLING_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# define function - Uses Open-Meteo's Weather Forecast API!!!
# Initialize geolocator
geolocator = Nominatim(user_agent="weather_app")

def get_weather(location: str):
    """
    Get detailed current, hourly, and 7-day weather forecast based on city name only.

    Args:
        location (str): City name (e.g., "Sydney", "New York")

    Returns:
        dict: Weather information or error.
    """
    # Geocode the city name
    location_obj = geolocator.geocode(location)
    if not location_obj:
        return {"error": "Location not found. Please provide a valid city name."}

    latitude = location_obj.latitude
    longitude = location_obj.longitude

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,wind_speed_10m,precipitation,rain,snowfall,weathercode"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,rain,snowfall,uv_index,cloudcover"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max,uv_index_max,sunshine_duration"
        f"&timezone=auto"
    )

    response = requests.get(url)
    data = response.json()

    return {
        "location": location,
        "coordinates": {"latitude": latitude, "longitude": longitude},
        "current": data.get("current", {}),
        "hourly": data.get("hourly", {}),
        "daily": data.get("daily", {})
    }

# define function - Uses FMG API!!!
def get_stock_data(symbol):
    api_key = os.getenv("FMG_API")  # Replace with your actual FMP API key
    base_url = "https://financialmodelingprep.com/api/v3"

    # Fetch stock price data in USD
    stock_url = f"{base_url}/quote/{symbol}?apikey={api_key}"
    stock_response = requests.get(stock_url)

    if stock_response.status_code != 200:
        return {"error": f"Stock API Error: {stock_response.status_code}"}

    try:
        stock_data = stock_response.json()
        if not stock_data or "price" not in stock_data[0]:
            return {"error": "Invalid stock symbol or no price data available."}

        stock_price_usd = stock_data[0]["price"]
        return {
            "symbol": symbol,
            "price_in_usd": stock_price_usd,
            "currency": "USD"  # Explicitly include the currency
        }
    except (IndexError, KeyError, ValueError) as e:
        return {"error": f"Error processing stock data: {str(e)}"}

# Define function to get cryptocurrency price using CoinGecko API
def get_crypto_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if symbol.lower() in data:
            return float(data[symbol.lower()]['usd'])  # Convert to float for accuracy
        else:
            return "Invalid cryptocurrency symbol or data unavailable"
    else:
        return f"Error: {response.status_code}, {response.text}"
#get_crypto_price("bitcoin")

def get_flight_info(flight_number):
    api_key = os.getenv("FLIGHT_API")  # Replace with your AviationStack API key
    url = f"http://api.aviationstack.com/v1/flights?access_key={api_key}&flight_iata={flight_number}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            flight_data = data["data"][0]  # Get the first matching flight

            flight_info = {
                "airline": flight_data.get("airline", {}).get("name", "Unknown"),
                "flight_number": flight_data.get("flight", {}).get("iata", "Unknown"),
                "departure_airport": flight_data.get("departure", {}).get("airport", "Unknown"),
                "departure_time": flight_data.get("departure", {}).get("estimated", "Unknown"),
                "arrival_airport": flight_data.get("arrival", {}).get("airport", "Unknown"),
                "arrival_time": flight_data.get("arrival", {}).get("estimated", "Unknown"),
                "status": flight_data.get("flight_status", "Unknown")
            }
            return flight_info
        else:
            return "Flight not found or invalid flight number"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example Usage:
#print(get_flight_info("AA100"))  # Replace with a real flight number

def get_exchange_rate(base_currency, target_currency):
    """
    Fetches the current exchange rate between two currencies.

    Parameters:
        base_currency (str): The base currency code (e.g., 'USD', 'EUR').
        target_currency (str): The target currency code (e.g., 'JPY', 'GBP').

    Returns:
        dict: A dictionary containing the exchange rate or an error message.
    """
    api_key = os.getenv("EXCHANGE_RATE_API")  # Replace with your actual ExchangeRate-API key
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "conversion_rate" in data:
            return {
                "base_currency": base_currency,
                "target_currency": target_currency,
                "exchange_rate": data["conversion_rate"]
            }
        else:
            return "Invalid currency pair or data unavailable"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example Usage:
#print(get_exchange_rate("USD", "EUR"))  # Fetch USD to EUR exchange rate

def convert_currency(base_currency, target_currency, amount):
    """
    Converts a specified amount from one currency to another using the ExchangeRate-API.
    """
    api_key = os.getenv("EXCHANGE_RATE_API")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "conversion_rate" in data:
            converted_amount = amount * data["conversion_rate"]
            return {
                "base_currency": base_currency,
                "target_currency": target_currency,
                "exchange_rate": data["conversion_rate"],
                "converted_amount": converted_amount
            }
        else:
            return "Invalid currency pair or data unavailable"
    else:
        return f"Error: {response.status_code}, {response.text}"

#print(convert_currency("USD", "EUR", 100))

def get_current_time_in_timezone(timezone):
    """
    Fetches the current time in the specified timezone.
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"timezone": timezone, "current_time": current_time}
    except pytz.UnknownTimeZoneError:
        return "Invalid timezone. Please provide a valid IANA timezone."

#print(get_current_time_in_timezone("America/Los_Angeles"))

def get_historical_stock_data(symbol, interval, limit=10):
    api_key = os.getenv("FMG_API")  # Replace with your actual FMP API key
    base_url = "https://financialmodelingprep.com/api/v3"

    # Fetch historical stock data
    historical_url = f"{base_url}/historical-chart/{interval}/{symbol}?apikey={api_key}"
    historical_response = requests.get(historical_url)

    if historical_response.status_code != 200:
        return {"error": f"Historical Data API Error: {historical_response.status_code}"}

    try:
        historical_data = historical_response.json()
        if not historical_data:
            return {"error": "No historical data available."}

        return {
            "symbol": symbol,
            "interval": interval,
            "data": historical_data[:limit],  # Return limited data
            "currency": "USD"
        }
    except (IndexError, KeyError, ValueError) as e:
        return {"error": f"Error processing historical data: {str(e)}"}


tools=[{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get detailed current, hourly, and 7-day weather forecast for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'Sydney', 'New York', 'Tokyo')."
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
  "type": "function",
  "function": {
    "name": "get_stock_data",
    "description": "Retrieve the latest/current stock price in USD for a given ticker symbol. For historical data, use 'get_historical_stock_data'.",
    "parameters": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "Stock ticker symbol (e.g., AAPL, TSLA)."
        }
      },
      "required": ["symbol"],
      "additionalProperties": False
    },
    "strict": True
  }
}, 
{
  "type": "function",
  "function": {
    "name": "get_historical_stock_data",
    "description": "Retrieve historical stock price data for a given ticker symbol. Use this if you need past prices or stock performance over a time range.",
    "parameters": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "Stock ticker symbol (e.g., AAPL, TSLA)."
        },
        "interval": {
          "type": "string",
          "description": "Historical data interval: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day, 1week, 1month."
        },
        "limit": {
          "type": "integer",
          "description": "Limit the number of historical data points. Default is 10."
        }
      },
      "required": ["symbol","interval","limit"],
      "additionalProperties": False
    },
    "strict": True
  }
},
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": "Fetches the current price of a cryptocurrency given its symbol (e.g., 'bitcoin', 'ethereum').",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"],
                "additionalProperties": False
            },
            "strict": True
        }   
    },
{
    "type": "function",
    "function": {
        "name": "get_flight_info",
        "description": "Fetches flight details including airline, departure, arrival, and status given a flight number (IATA format).",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "IATA flight number (e.g., 'AA100', 'DL405')."
                }
            },
            "required": ["flight_number"],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "Fetches the exchange rate between two currencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "base_currency": {
                    "type": "string",
                    "description": "The base currency code (e.g., 'USD', 'EUR')."
                },
                "target_currency": {
                    "type": "string",
                    "description": "The target currency code (e.g., 'JPY', 'GBP')."
                }
            },
            "required": ["base_currency", "target_currency"],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "get_current_time_in_timezone",
        "description": "Fetches the current time in a specified timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone format (e.g., 'America/New_York', 'Asia/Tokyo')."}
            },
            "required": ["timezone"],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Converts a specified amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "base_currency": {"type": "string", "description": "Base currency code (e.g., 'USD')."},
                "target_currency": {"type": "string", "description": "Target currency code (e.g., 'AUD')."},
                "amount": {"type": "number", "description": "Amount to be converted."}
            },
            "required": ["base_currency", "target_currency", "amount"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

# Define the endpoint
@app.post("/", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    # Get user input
    user_input = request.user_input
    
    # Create a new message list for this request only
    messages = [{"role": "user", "content": user_input}]
    
    # Get the AI's response
    completion = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
        response_format={
            "type": "text"
        },
        tools=tools,
        temperature=1,
        top_p=1,
        presence_penalty=0
    )
    
    # Check if the AI's response includes a tool call
    if completion.choices[0].message.tool_calls:
        tool_call = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        if completion.choices[0].message.tool_calls[0].function.name == "get_weather":
            # Call the function with the provided arguments
            result = get_weather(args["location"])
        
        elif completion.choices[0].message.tool_calls[0].function.name == "get_stock_data":
        # Call the function with the provided arguments
            result = get_stock_data(args["symbol"])

        elif completion.choices[0].message.tool_calls[0].function.name == "get_crypto_price":
            # Call the function with the provided arguments
            result = get_crypto_price(args["symbol"])

        elif completion.choices[0].message.tool_calls[0].function.name == "get_flight_info":
            # Call the function with the provided arguments
            result = get_flight_info(args["flight_number"])

        elif completion.choices[0].message.tool_calls[0].function.name == "get_exchange_rate":
            # Call the function with the provided arguments
            result = get_exchange_rate(args["base_currency"], args["target_currency"])

        elif completion.choices[0].message.tool_calls[0].function.name == "get_current_time_in_timezone":
            # Call the function with the provided arguments
            result = get_current_time_in_timezone(args["timezone"])

        elif completion.choices[0].message.tool_calls[0].function.name == "convert_currency":
            # Call the function with the provided arguments
            result = convert_currency(args["base_currency"], args["target_currency"], args["amount"])
        
        elif completion.choices[0].message.tool_calls[0].function.name == "get_historical_stock_data":
            symbol = args.get("symbol")
            interval = args.get("interval", "1day")
            limit = args.get("limit", 10)  # Default to 10 if not provided

            # Detect if this is an incorrect use of the historical function
            if "current" in user_input.lower() or "latest" in user_input.lower():
                # If it seems like the user wanted current price, call the correct function
                result = get_stock_data(symbol)
            else:
                result = get_historical_stock_data(symbol, interval, limit)
                result = truncate_response(result)

        # Append the tool call and result to the messages
        messages.append({
            "role": "assistant",
            "content": [],
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }]
        })
        messages.append({
            "role": "tool",
            "content": [{
                "type": "text",
                "text": json.dumps(result)
            }],
            "tool_call_id": tool_call.id
        })
        
        # Get the final AI message after incorporating the tool result
        completion = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages,
            tools=tools
        )
    
    # Extract the AI's message content
    ai_message_content = completion.choices[0].message.content
    
    response_data = {
        "ai_response": ai_message_content,
        "function_called": None,
        "function_args": None,
        "function_response": None
    }

    # If there was a function call, include the details
    if len(messages) > 1 and "tool_calls" in messages[1]:
        tool_call = messages[1]["tool_calls"][0]
        response_data.update({
            "function_called": tool_call["function"]["name"],
            "function_args": json.loads(tool_call["function"]["arguments"]),
            "function_response": json.loads(messages[2]["content"][0]["text"])
        })
    
    return response_data

# Run the app with: uvicorn gemini_function_calling:app --reload