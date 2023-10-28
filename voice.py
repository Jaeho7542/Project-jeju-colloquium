import openai
import time, os
import speech_recognition as sr
from gtts import gTTS
import playsound


def generate_chat(question: str) -> str:
    """gpt로 그림 관련 문장을 생성하는 함수"""
    
    openai.api_key = "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"
    
    history_message = [
    {"role": "system", "content":"너는 OOO 미술관을 설명하는 가이드 인공지능이야."},
    {"role": "system", "content": "미술관에는 모나리자, 별이 빛나는 밤, 절규, 키스, 진주 귀고리를 한 소녀가 있어."} 
                        ] # You are an AI assistant that explains items in art galleries recognized by the user
    
    history_message.append({"role":"user", "content":question})
    
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history_message
    )
    
    message = completions.choices[0].message.to_dict()
    answer = message["content"].strip()
    history_message.append(message)
    
    return answer

def text_to_speech1(texts):
    filename = "check.mp3"
    tts = gTTS(text=texts, lang='ko')
    tts.save(filename)
    playsound.playsound(filename)
    if os.path.exists(filename):
        os.remove(filename)
        
def text_to_speech2(texts):
    filename = "answer.mp3"
    tts = gTTS(text=texts, lang='ko')
    tts.save(filename)
    playsound.playsound(filename)
    if os.path.exists(filename):
        os.remove(filename)

def speech_to_text(r, audio):
    try:
        print('듣는 중입니다.')
        text_to_speech1("듣는 중입니다")
        text_p = r.recognize_google(audio, language="ko-KR")
        print("[사람]" + ":" + text_p)        
        text_g = generate_chat(text_p)
        print("GPT" + ":" + text_g)
        text_to_speech2(text_g)
            
    except sr.UnknownValueError:
        return "Error 1"
    except sr.RequestError:
        return "Error 2"

def talking() -> None:
    
    r = sr.Recognizer()
    m = sr.Microphone()
    
    r.listen_in_background(m, speech_to_text)
    
    while True:
        time.sleep(0.5)


# text_to_speech("안녕하세요")