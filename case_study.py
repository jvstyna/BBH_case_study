import yfinance as yf
from datetime import datetime
from transformers import pipeline

ASSETS = ["AAPL", "TSLA", "AMZN", "^GSPC", "SPY"]

def get_client_info():
    """ 
    Gets client information such as name, age, investment horizon,
    and current savings amount.

    Returns:
        client (dict): Dictionary with client information.
    """
    client={}
    print("Enter the client's name:")
    client["name"] = str(input())
    print("Enter the client's age:")
    client["age"] = int(input())
    print("Enter the client's investment horizon (in years):")
    client["investment_horizon"] = int(input())
    print("Enter the client's current savings amount (in dollars):")
    client["savings"] = float(input())
    return client


def get_stock_data():
    """
    Gets stock data from Yahoo Finance and calculates annualized returns.

    Returns:
        stock_data (dict): Dictionary with asset names as keys and their
        annualized returns as values.
    """
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    stock_data={}
    market_data = yf.download(ASSETS, start, end)['Close']
    for asset in market_data.columns:
        daily_return = market_data[asset].pct_change().mean()
        annualized_return = daily_return * 356
        stock_data[asset] = annualized_return
    return stock_data


def generate_investment_plan():
    """
    Generates an investment plan using a generative language model
    from the Transformers library.

    Args:
        stock_data (dict): Dictionary with asset names as keys and their
        annualized returns as values.

    Returns:
        str: Generated investment plan.
    """
    generator = pipeline("text-generation", model="openai-community/gpt2")
    client_info = get_client_info()
    prompt = ("Based on the above data, generate a clear and concise investment plan tailored for the client,balancing risk and growth opportunities while considering the client's goals:"
        f"Client Name: {client_info['name']}\n"
        f"Age: {client_info['age']} years\n"
        f"Investment Horizon: {client_info['investment_horizon']} years\n"
        f"Current Savings: {client_info['savings']}\n"
        "Annualized Returns for Assets:\n")

    for asset, annualized_return in get_stock_data().items():
        prompt += f"{asset}: {annualized_return:.2f}%\n"
    result = generator(prompt, max_length=300, num_return_sequences=1)

    return result[0]['generated_text']


if __name__ == "__main__":
    print(generate_investment_plan())