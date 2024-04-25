from fastapi import FastAPI, UploadFile, File, Query
from tempfile import NamedTemporaryFile
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from generate_qa import generate_questions_and_answers

app = FastAPI()

# openai api key required if using openai embeddings
os.environ["OPENAI_API_KEY"] = ""


def create_docs(pdf_path, embedding_model, max_chunk_length):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if embedding_model == "spacy":
        embedding_model_instance = SpacyEmbeddings(model_name="en_core_web_sm")
    elif embedding_model == "openai":
        embedding_model_instance = OpenAIEmbeddings()
    else:
        raise ValueError("Invalid embedding model specified. Choose 'spacy' or 'openai'.")

    text_splitter = SemanticChunker(embedding_model_instance)

    modified_docs = []
    for doc in documents:
        content = doc.page_content
        if len(content) > max_chunk_length:
            chunks = [content[i:i + max_chunk_length] for i in range(0, len(content), max_chunk_length)]
            new_docs = text_splitter.create_documents(chunks)
            modified_docs.extend(new_docs)
        else:
            modified_docs.append(doc)

    return modified_docs


def print_sorted_qa_by_confidence(all_q_and_a, num_results):
    sorted_q_and_a = sorted(all_q_and_a, key=lambda x: x['confidence'], reverse=True)

    if num_results > len(sorted_q_and_a):
        raise ValueError("Number of results requested exceeds the available data")

    result = []
    for idx, qa in enumerate(sorted_q_and_a[:num_results], 1):
        qa_result = {
            "question": qa['question'],
            "confidence": qa['confidence'],
            "options": qa['options'],
            "answer": qa['answer']
        }
        result.append(qa_result)

    return result


@app.post("/generate_qa/")
async def generate_qa(
    pdf: UploadFile = File(...),
    embedding_model: str = Query(..., regex="^(spacy|openai)$"),
    max_chunk_length: int = Query(..., gt=0),
    min_chunk_length: int = Query(..., gt=0),
    num_results: int = Query(..., gt=0)
):
    with NamedTemporaryFile(delete=False) as tmp_pdf:
        tmp_pdf.write(await pdf.read())
        tmp_pdf_path = tmp_pdf.name

    try:
        modified_docs = create_docs(tmp_pdf_path, embedding_model, max_chunk_length)

        all_q_and_a = []

        for doc in modified_docs:
            content_length = len(doc.page_content)
            print("Length of content:", content_length)

            if content_length < min_chunk_length:
                continue

            num_questions = 2
            q_and_a = generate_questions_and_answers(doc.page_content, num_questions)
            all_q_and_a.extend(q_and_a)

        sorted_qa = print_sorted_qa_by_confidence(all_q_and_a, num_results)

        return sorted_qa
    finally:
        os.unlink(tmp_pdf_path)
