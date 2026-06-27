import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def build_and_save_index(self) -> None:
        kb_path = Path(config.KB_DIR)
        if not kb_path.exists() or not any(kb_path.glob("*.md")):
            logger.warning("Knowledge base directory is empty or missing.")
            return

        documents = []
        metadata_list = []

        for file_path in kb_path.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append(content)
                metadata_list.append({"source": file_path.name})
            except Exception as e:
                logger.error(f"Could not read {file_path}: {e}")

        if not documents:
            logger.error("No markdown files found to index.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )

        split_docs = []
        split_metadatas = []
        for doc, meta in zip(documents, metadata_list):
            for chunk in text_splitter.split_text(doc):
                split_docs.append(chunk)
                split_metadatas.append(meta)

        logger.info(f"Building FAISS index with {len(split_docs)} chunks...")
        self.vector_store = FAISS.from_texts(
            texts=split_docs,
            embedding=self.embeddings,
            metadatas=split_metadatas
        )
        self.vector_store.save_local(str(config.FAISS_INDEX_PATH))
        logger.info(f"Index saved to {config.FAISS_INDEX_PATH}")

    def load_index(self) -> bool:
        index_path = Path(config.FAISS_INDEX_PATH) / "index.faiss"
        if not index_path.exists():
            logger.warning("Index not found. Attempting to build from scratch...")
            self.build_and_save_index()
            if not index_path.exists():
                logger.error("Index build failed.")
                return False

        try:
            self.vector_store = FAISS.load_local(
                str(config.FAISS_INDEX_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def retrieve_chunks(self, query: str, k: int = config.TOP_K) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            if not self.load_index():
                return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            chunks = []
            for doc, distance in results:
                # Convert L2 distance to cosine similarity: similarity = 1 - (d^2 / 2)
                similarity = max(0.0, min(1.0, 1.0 - (distance ** 2) / 2.0))
                chunks.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "distance": float(distance),
                    "similarity": float(similarity)
                })
            return chunks
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []


if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever()
    retriever.build_and_save_index()
    retriever.load_index()
    results = retriever.retrieve_chunks("What documents are required to register as Eko partner?")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['source']} | similarity: {r['similarity']:.4f}")
        print(r['content'][:150])
