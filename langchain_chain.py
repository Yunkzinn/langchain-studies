from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv  

load_dotenv()
set_debug(True)

modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu insteresse por {interesse}. A sua saída deve ser SOMENTE o nome da cidade. Cidade:" # Não funciona sempre, melhor formatar sempre
)

modelo_resturantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

llm = ChatOpenAI(
    model="",
    temperature="",
    api_key=os.getenv("OPENAI_API_KEY"),
    )

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes = LLMChain(prompt=modelo_resturantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(chains=[cadeia_cidade, cadeia_restaurantes, cadeia_cultural], verbose=True)

resultado = cadeia.invoke("praias")
print(resultado)
