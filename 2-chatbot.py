from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_ID_LEGACY_FASTEST="gemini-2.5-flash-lite"
MODEL_ID_LEGACY_FAST="gemini-2.5-flash"
MODEL_ID_LEGACY_DEEP="gemini-2.5-pro"
MODEL_ID_FASTEST= "gemini-3.1-flash-lite-preview"
MODEL_ID_FAST="gemini-3.1-flash-preview"
MODEL_ID_DEEP="gemini-3.1-pro-preview"
MODEL_GEMMA4="gemma-4-26b-a4b-it"

# Inicializa a sessão de chat para manter o contexto automaticamente
chat = client.chats.create(
    model=MODEL_GEMMA4, 
    config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1024
    )
)

def enviar_mensagem(texto):
    """Envia mensagem e processa a resposta em tempo real (streaming)."""
    try:
        resposta = chat.send_message_stream(texto)
        print("Chatbot: ", end="", flush=True)
        for chunk in resposta:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nErro na API: {e}")

if __name__ == "__main__":
    print("--- Chatbot Ativo (digite 'sair' para parar) ---")

    while True:
        user_input = input("Você: ")
        
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Encerrando...")
            break
            
        if user_input.strip():
            enviar_mensagem(user_input)
