import os
from google import genai
from google.genai import types
import speech_recognition as sr
from playsound import playsound
from pathlib import Path
from io import BytesIO

# Inicialização do cliente Google GenAI
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

arquivo_audio = "hello.mp3"

recognizer = sr.Recognizer()

def grava_audio():
    """Captura áudio do microfone e retorna o áudio gravado"""
    with sr.Microphone(0) as source:
        print("Ouvindo...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    return audio

def transcricao_audio(audio):
    """Transcreve o áudio utilizando o modelo Gemini"""
    try:
        wav_data = audio.get_wav_data()
        # O Gemini 2.0 Flash pode processar áudio diretamente para transcrição
        resposta = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                "Transcreva este áudio."
            ]
        )
        return resposta.text
    except Exception as e:
        print(f"Erro na transcrição do áudio: {e}")
        return ""
    
def completa_texto(mensagens):
    """Gera uma resposta com base no histórico de mensagens usando Gemini"""
    try:
        resposta = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=mensagens,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0
            )
        )
        return resposta.text
    except Exception as e:
        print(f"Erro na geração de resposta: {e}")
        return "Desculpe, não consegui entender"
    
def cria_audio(texto):
    """Cria um arquivo de áudio a partir do texto usando a capacidade nativa do Gemini"""
    if Path(arquivo_audio).exists():
        Path(arquivo_audio).unlink()
    try:
        # Gerando áudio nativamente com Gemini 2.0
        resposta = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=texto,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"]
            )
        )
        # Extraindo os bytes do áudio da resposta
        for part in resposta.candidates[0].content.parts:
            if part.inline_data:
                with open(arquivo_audio, "wb") as f:
                    f.write(part.inline_data.data)
                return
    except Exception as e:
        print(f"Erro na criação de áudio: {e}")
        
def roda_audio():
    """Reproduz o arquivo de áudio gerado"""
    if Path(arquivo_audio).exists():
        playsound(arquivo_audio)
    else:
        print("Erro: O arquivo de áudio não foi encontrado.")
        
def main():
    """Função principal para executar o assistente de voz"""
    mensagens = []
    while True:
        audio = grava_audio()
        transcricao = transcricao_audio(audio)
        
        if not transcricao:
            print("Não foi possível transcrever o áudio. Tente novamente")
            continue
    
        print(f"User: {transcricao}")
        # Formato de mensagens esperado pelo SDK google-genai
        mensagens.append({"role": "user", "parts": [{"text": transcricao}]})
        
        resposta_texto = completa_texto(mensagens)
        print(f"Assistant: {resposta_texto}")
        mensagens.append({"role": "model", "parts": [{"text": resposta_texto}]})
        
        cria_audio(resposta_texto)
        roda_audio()

if __name__ == "__main__":
    main()
