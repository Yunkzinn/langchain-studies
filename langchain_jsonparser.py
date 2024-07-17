from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv  

load_dotenv()
set_debug(True)

class Destino:
    cidade = Field("Cidade a visitar")
    motivo = Field("Motivo pela qual é interessante visitar")

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template="""Sugira uma cidade dado meu insteresse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions},
)

modelo_resturantes = PromptTemplate(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = PromptTemplate(
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
