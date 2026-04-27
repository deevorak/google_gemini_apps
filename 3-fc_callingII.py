import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa o cliente do Gemini
# Certifique-se de que a variável GEMINI_API_KEY esteja definida no seu .env
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Modelo solicitado
MODEL_GEMMA4 = "gemma-4-26b-a4b-it"

# Função para calcular o IMC e fornecer recomendação
def calcular_imc(peso: float, altura: float) -> str:
    """Calcula o IMC de uma pessoa e fornece uma recomendação de saúde.
    
    Args:
        peso: Peso da pessoa em kg.
        altura: Altura da pessoa em metros.
    """
    imc = peso / (altura ** 2)
    if imc < 18.5:
        estado = "abaixo do peso"
        recomendacao = "É importante que você consulte um médico para ajustar sua alimentação."
    elif 18.5 <= imc < 24.9:
        estado = "peso normal"
        recomendacao = "Continue mantendo hábitos saudáveis!"
    elif 25 <= imc < 29.9:
        estado = "sobrepeso"
        recomendacao = "Você pode considerar uma reavaliação de sua dieta e exercícios físicos."
    else:
        estado = "obesidade"
        recomendacao = "É altamente recomendável consultar um médico para orientações sobre perda de peso."
    
    return json.dumps({
        "imc": round(imc, 2),
        "estado": estado,
        "recomendacao": recomendacao
    })

# Lista de ferramentas (funções) disponíveis para o modelo
tools = [calcular_imc]

# Inicializa uma sessão de chat com suporte a ferramentas
# O SDK google-genai gerencia chamadas de função automaticamente em sessões de chat
chat = client.chats.create(
    model=MODEL_GEMMA4,
    config=types.GenerateContentConfig(
        tools=tools,
        temperature=0
    )
)

mensagem = "Qual é o IMC de uma pessoa que pesa 70 kg e tem 1.75 m de altura?"

print(f"Usuário: {mensagem}")

# Envia a mensagem. O SDK detectará a necessidade de chamar calcular_imc,
# executará a função localmente e enviará o resultado de volta para o modelo.
try:
    resposta = chat.send_message(mensagem)
    print(f"\nGemini ({MODEL_GEMMA4}):")
    print(resposta.text)
except Exception as e:
    print(f"\nErro ao chamar a API: {e}")
