from pathlib import Path
import chromadb
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer
import re

DATA_DIR = Path("data")
OUT_DIR = Path("markdown")
CHROMA_DIR = Path("chroma_db")

PDFS = [
    ("2025-Q1", "reporteEneroMarzo2025.pdf"),
    ("2025-Q2", "reporteAbrilJunio2025.pdf"),
    ("2025-Q3", "reporteJulioSeptiembre2025.pdf"),
    ("2025-Q4", "reporteOctubreDiciembre2025.pdf"),
]

def pdfs_to_markdown():
    OUT_DIR.mkdir(exist_ok=True)
    md = MarkItDown()

    markdown_files = []

    for quarter, filename in PDFS:
        pdf_path = DATA_DIR / filename
        md_path = OUT_DIR / f"{quarter}.md"

        print(f"Convirtiendo {pdf_path}...")
        result = md.convert(str(pdf_path))

        md_path.write_text(result.text_content, encoding="utf-8")
        markdown_files.append((quarter, md_path))

    return markdown_files

def clean_markdown(text):
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Quita separadores de tablas tipo | --- | --- |
        if re.fullmatch(r"[\|\-\s:]+", stripped):
            continue

        # Quita líneas que parecen tablas muy fragmentadas
        if stripped.count("|") >= 6:
            stripped = stripped.replace("|", " ")

        # Normaliza espacios
        stripped = re.sub(r"\s+", " ", stripped).strip()

        if stripped:
            cleaned.append(stripped)

    return "\n".join(cleaned)

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks

def build_index(markdown_files):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="reportes_2025")

    ids = []
    documents = []
    metadatas = []

    for quarter, md_path in markdown_files:
        text = md_path.read_text(encoding="utf-8")
        text = clean_markdown(text)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            ids.append(f"{quarter}-{i}")
            documents.append(chunk)
            metadatas.append({
                "quarter": quarter,
                "source": md_path.name,
                "chunk": i,
            })

    print("Generando embeddings...")
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexados {len(documents)} chunks en {CHROMA_DIR}")

def search(query, top_k=5):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="reportes_2025")

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        print("\n" + "=" * 80)
        print(f"Resultado {i + 1}")
        print(f"Quarter: {metadata['quarter']}")
        print(f"Fuente: {metadata['source']}")
        print(f"Chunk: {metadata['chunk']}")
        print("-" * 80)
        print(doc[:1200])

if __name__ == "__main__":
    markdown_files = pdfs_to_markdown()
    build_index(markdown_files)

    print("\nPrueba de búsqueda:")
    search("¿Cuáles fueron los principales resultados del año?")