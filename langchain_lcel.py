from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from dotenv import load_dotenv  

load_dotenv()
set_debug(True)

class Destino:
    cidade = Field("Cidade a visitar")
    motivo = Field("Motivo pela qual é interessante visitar")

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = ChatPromptTemplate.from_template(
    template="""Sugira uma cidade dado meu insteresse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions},
)

modelo_resturantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

modelo_final = ChatPromptTemplate.from_messages(
    [
    ("ai", "Sugestão de viagem para a cidade: {cidade}"),
    ("ai", "Restaurantes que você não pode perder: {restaurantes}"),
    ("ai", "Atividades e locais culturais recomendados: {locais_culturais}"),
    ("system", "Combine as informações anteriores em 2 parágrafos coerentes"),
    ]
)

llm = ChatOpenAI(
    model="",
    temperature="",
    api_key=os.getenv("OPENAI_API_KEY"),
    )

part1 = modelo_cidade | llm | parseador
part2 = modelo_resturantes | llm | StrOutputParser()
part3 = modelo_cultural | llm | StrOutputParser()
part4 = modelo_final | llm | StrOutputParser()

cadeia = (part1 | {
    "restaurantes": part2, 
    "locais_culturais": part3,
    "cidade" : itemgetter("cidade")
    }
    | part4)

resultado = cadeia.invoke({"interesse" : "praias"})
print(resultado)