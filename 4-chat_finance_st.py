import streamlit as st
import os
import yfinance as yf
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

# Configuração do cliente Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = "gemma-4-26b-a4b-it" # Ou outro modelo de sua preferência

# Função para buscar cotações
def retorna_cotacao(ticker: str, periodo: str = "1mo") -> str:
    """
    Retorna a cotação de ações da Ibovespa.

    Args:
        ticker: O ticker da ação. Ex: BBAS3, ITSA4, etc.
        periodo: Período dos dados históricos. Opções: '1d', '5d', '1mo', '6mo', '1y', '5y', '10y', 'ytd', 'max'.
    """
    ticker_obj = yf.Ticker(f"{ticker}.SA")
    hist = ticker_obj.history(period=periodo)["Close"]
    hist.index = hist.index.map(lambda x: x.strftime("%Y-%m-%d"))
    hist = round(hist, 2)
    if len(hist) > 30:  # Limita a 30 amostras
        slice_size = int(len(hist) / 30)
        hist = hist.iloc[::-slice_size][::-1]
    return hist.to_json()

# Ferramentas disponíveis
tools = [retorna_cotacao]

# Configuração da interface do Streamlit
st.set_page_config(page_title="Chatbot com Ações", page_icon="🤖")

# Título
st.title("Chatbot de Cotações de Ações 📈")

# Inicializa o estado das mensagens
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibição do histórico de mensagens
for msg in st.session_state.mensagens:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).markdown(msg["content"])

# Entrada de mensagem do usuário
user_input = st.chat_input("Digite sua pergunta sobre cotações de ações...")

if user_input:
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.mensagens.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)
    
    # Inicializa o chat com o histórico existente
    # O SDK google-genai espera o histórico no formato [{'role': 'user', 'parts': [{'text': '...'}]}, ...]
    history = []
    for msg in st.session_state.mensagens[:-1]: # Não inclui a última que acabou de ser enviada
        history.append(types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part(text=msg["content"])]
        ))

    chat = client.chats.create(
        model=MODEL_ID,
        config=types.GenerateContentConfig(
            tools=tools,
            temperature=0
        ),
        history=history
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Pensando...")
        
        try:
            # send_message com o automático function calling ativado no chat
            response = chat.send_message(user_input)
            
            # Adiciona a resposta do chatbot ao histórico
            st.session_state.mensagens.append({"role": "model", "content": response.text})
            placeholder.markdown(response.text)
            
        except Exception as e:
            error_msg = f"Erro ao processar sua solicitação: {e}"
            st.error(error_msg)
            st.session_state.mensagens.append({"role": "model", "content": error_msg})
