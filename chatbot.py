import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "reportes_2025"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


RAG_SYSTEM_PROMPT = """
Eres un asistente RAG para consultar los Informes Trimestrales 2025.

Reglas:
- Responde únicamente con base en el contexto proporcionado.
- Si el contexto no contiene suficiente información, dilo claramente.
- Responde en español.
- Sé claro y breve.
"""


NO_RAG_SYSTEM_PROMPT = """
Eres un asistente general.

Reglas:
- Responde en español.
- Sé claro y breve.
- No inventes que consultaste documentos, archivos o fuentes.
- Si no tienes información suficiente, dilo claramente.
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        (
            "human",
            """
            Pregunta:
            {question}

            Contexto recuperado:
            {context}
            """,
        ),
    ]
)


NO_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NO_RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def load_vectorstore():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No existe {CHROMA_DIR}. Primero ejecuta rag_index.py para crear el índice."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        quarter = metadata.get("quarter", "desconocido")
        source = metadata.get("source", "desconocido")
        chunk = metadata.get("chunk", "desconocido")

        formatted.append(
            f"[Documento {i}]\n"
            f"Trimestre: {quarter}\n"
            f"Fuente: {source}\n"
            f"Chunk: {chunk}\n"
            f"Contenido:\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def print_retrieved_chunks(docs):
    print("\nChunks recuperados:\n")

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        quarter = metadata.get("quarter", "desconocido")
        source = metadata.get("source", "desconocido")
        chunk = metadata.get("chunk", "desconocido")
        preview = doc.page_content.replace("\n", " ")[:250]

        print(f"{i}. Trimestre: {quarter} | Fuente: {source} | Chunk: {chunk}")
        print(f"   Preview: {preview}...")


def ask_with_rag(question, vectorstore, llm, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    return answer, docs


def ask_without_rag(question, llm):
    chain = NO_RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question})

    return answer


def print_help():
    print("Comandos disponibles:")
    print("  /rag       Activa respuestas usando Chroma + documentos recuperados")
    print("  /norag     Activa respuestas directas del modelo, sin documentos")
    print("  /compare   Responde la misma pregunta en modo sin RAG y con RAG")
    print("  /help      Muestra estos comandos")
    print("  salir      Termina el programa")


def main():
    print("Cargando índice vectorial...")
    vectorstore = load_vectorstore()

    print(f"Cargando modelo Ollama: {OLLAMA_MODEL}")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    mode = "rag"

    print("\nChatbot listo. Modo inicial: RAG.")
    print_help()
    print()

    while True:
        question = input("Pregunta: ").strip()

        if question.lower() in {"salir", "exit", "quit", "q"}:
            print("Hasta luego.")
            break

        if not question:
            continue

        if question == "/help":
            print_help()
            print()
            continue

        if question == "/rag":
            mode = "rag"
            print("Modo activo: RAG\n")
            continue

        if question == "/norag":
            mode = "norag"
            print("Modo activo: sin RAG\n")
            continue

        if question == "/compare":
            comparison_question = input("Pregunta para comparar: ").strip()

            if not comparison_question:
                continue

            try:
                print("\nRespuesta SIN RAG:\n")
                print(ask_without_rag(comparison_question, llm))

                print("\n" + "-" * 80)
                print("\nRespuesta CON RAG:\n")
                answer, docs = ask_with_rag(comparison_question, vectorstore, llm)
                print(answer)
                print_retrieved_chunks(docs)

                print("\n" + "=" * 80 + "\n")
            except Exception as exc:
                print(f"\nError: {exc}\n")

            continue

        try:
            docs = None

            if mode == "rag":
                answer, docs = ask_with_rag(question, vectorstore, llm)
                label = "Respuesta CON RAG"
            else:
                answer = ask_without_rag(question, llm)
                label = "Respuesta SIN RAG"

            print(f"\n{label}:\n")
            print(answer)

            if docs is not None:
                print_retrieved_chunks(docs)

            print("\n" + "=" * 80 + "\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()