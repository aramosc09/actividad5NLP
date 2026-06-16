# actividad5NLP — Chatbot con RAG y LLM

Archivo principal para la entrega de la actividad: `MNA_NLP_actividad_chatbot_LLM_RAG_R.ipynb`

Chatbot que responde preguntas sobre los **Informes Trimestrales 2025** combinando un
LLM local (vía [Ollama](https://ollama.com/)) con **Recuperación Aumentada Generativa
(RAG)** sobre una base vectorial [Chroma](https://www.trychroma.com/).

El proyecto cubre las dos etapas típicas de un sistema RAG:

1. **Indexado** ([rag_index.py](rag_index.py)): convierte los PDF a Markdown, los limpia,
   los divide en *chunks*, genera *embeddings* y los guarda en Chroma.
2. **Consulta** ([chatbot.py](chatbot.py)): recupera los *chunks* más relevantes y los
   pasa al LLM para generar una respuesta fundamentada. Incluye un modo de comparación
   con/sin RAG.

> Este es un primer avance del equipo. La lógica base de indexado y chatbot ya funciona;
> el resto del trabajo (mejoras de prompts, evaluación, interfaz, etc.) queda pendiente.

## Contexto

Las organizaciones enfrentan el reto de gestionar grandes volúmenes de información y
ofrecer respuestas precisas y contextualizadas. Los chatbots basados en LLM con RAG
combinan la generación de texto de los modelos de lenguaje con la recuperación de
documentos específicos, lo que permite respuestas más precisas y fundamentadas. Su
aplicación es clave en atención al cliente, educación, salud y gestión del conocimiento.

## Arquitectura

```
data/*.pdf  ──>  markitdown  ──>  markdown/*.md  ──>  limpieza + chunking
                                                            │
                                                            ▼
                                          sentence-transformers (embeddings)
                                                            │
                                                            ▼
                                                   chroma_db/ (Chroma)
                                                            │
              pregunta  ──>  retriever (top-k)  ──────────────┘
                                  │
                                  ▼
                         ChatPromptTemplate  ──>  ChatOllama (LLM)  ──>  respuesta
```

- **Modelo de embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (usado tanto al
  indexar como al consultar).
- **Colección Chroma:** `reportes_2025`, persistida en `chroma_db/`.
- **LLM:** servido por Ollama (configurable, ver más abajo).

## Requisitos previos

- **Python 3.10+**
- **[Ollama](https://ollama.com/) instalado y en ejecución** con un modelo de chat
  descargado. Por ejemplo:
  ```bash
  ollama pull llama3.2
  ```
- Los PDF de los reportes en la carpeta `data/` con estos nombres (ver lista en
  [rag_index.py](rag_index.py#L11-L16)):
  - `reporteEneroMarzo2025.pdf` (2025-Q1)
  - `reporteAbrilJunio2025.pdf` (2025-Q2)
  - `reporteJulioSeptiembre2025.pdf` (2025-Q3)
  - `reporteOctubreDiciembre2025.pdf` (2025-Q4)

> Nota: `data/`, `markdown/` y `chroma_db/` están en `.gitignore`, así que no vienen en
> el repo. Cada quien debe colocar sus PDF en `data/` y generar el índice localmente.

## Instalación

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Configuración

El modelo de Ollama se elige con la variable de entorno `OLLAMA_MODEL`
(por defecto `llama3.2`, ver [chatbot.py](chatbot.py#L14)):

```bash
export OLLAMA_MODEL=llama3.2     # o el modelo que tengas descargado en Ollama
```

## Uso

### 1. Construir el índice vectorial

Convierte los PDF, genera los *embeddings* y crea la base Chroma. Ejecútalo una vez
(o cada vez que cambien los documentos):

```bash
python rag_index.py
```

Esto crea las carpetas `markdown/` y `chroma_db/`, y al final hace una búsqueda de prueba.

### 2. Iniciar el chatbot

```bash
python chatbot.py
```

Arranca en modo RAG. Comandos disponibles dentro del chat:

| Comando     | Acción                                                            |
|-------------|-------------------------------------------------------------------|
| `/rag`      | Respuestas usando Chroma + documentos recuperados                 |
| `/norag`    | Respuestas directas del modelo, sin documentos                    |
| `/compare`  | Responde la misma pregunta sin RAG y con RAG, para comparar       |
| `/help`     | Muestra los comandos                                              |
| `salir`     | Termina el programa (`exit`, `quit`, `q` también funcionan)       |

En modo RAG el chatbot también imprime los *chunks* recuperados (trimestre, fuente y
vista previa) para facilitar la verificación de las respuestas.

## Estructura del proyecto

```
.
├── chatbot.py        # CLI del chatbot (RAG / sin RAG / comparación)
├── rag_index.py      # Pipeline de indexado: PDF -> Markdown -> Chroma
├── requirements.txt  # Dependencias de Python
├── data/             # PDFs de entrada (no versionado)
├── markdown/         # Markdown generado (no versionado)
├── chroma_db/        # Base vectorial Chroma (no versionado)
└── COMPARE.md        # Notas de comparación con/sin RAG
```

## Solución de problemas

- **`No existe chroma_db ...`**: ejecuta primero `python rag_index.py`.
- **Errores de conexión con el LLM**: verifica que Ollama esté corriendo
  (`ollama list`) y que el modelo de `OLLAMA_MODEL` esté descargado.
- **No encuentra los PDF**: confirma que estén en `data/` con los nombres exactos
  esperados en [rag_index.py](rag_index.py#L11-L16).
