import yfinance as yf
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configuração do cliente Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = "gemma-4-26b-a4b-it"

def retorna_cotacao(ticker: str, periodo: str = "1mo") -> str:
    """
    Retorna a cotação de ações da Ibovespa.

    Args:
        ticker: O ticker da ação. Ex: BBAS3, BBDC4, etc.
        periodo: Período retornado dos dados históricos da cotação. Opções: '1d', '5d', '1mo', '6mo', '1y', '5y', '10y', 'ytd', 'max'.
    """
    ticker_obj = yf.Ticker(f"{ticker}.SA")
    hist = ticker_obj.history(period=periodo)["Close"]
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist = round(hist, 2)
    # Limitar em 30 resultados
    if len(hist) > 30:
        slice_size = int(len(hist) / 30)
        hist = hist.iloc[::-slice_size][::-1]
    return hist.to_json()

# Ferramentas disponíveis
tools = [retorna_cotacao]

# Inicializa o chat com ferramentas
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        tools=tools,
        temperature=0
    )
)

mensagem = "Qual é a cotação da Vale no último ano?"
print(f"Usuário: {mensagem}")

try:
    # send_message lida com a chamada de função automaticamente
    resposta = chat.send_message(mensagem)
    print(f"\nGemini: {resposta.text}")
except Exception as e:
    print(f"\nErro ao processar: {e}")
