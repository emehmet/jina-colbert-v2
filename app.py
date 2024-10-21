from pydantic import BaseModel, ValidationError
from ragatouille import RAGPretrainedModel
import faiss


from flask import Flask, request, jsonify


import os
import threading
from ragatouille import RAGPretrainedModel
import json



# Önceden eğitilmiş model ismi
pretrained_model_name = "jinaai/jina-colbert-v2"
index_path = "/home/ec2-user/pyton-projects/jina-colbert-v2/"
RAG = RAGPretrainedModel.from_pretrained(pretrained_model_name,index_root=index_path)
# Kaydetmek istediğin modelin path'i

app = Flask(__name__)
# Update any base URLs to use the public ngrok URL

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 16 MB

print("Model loaded successfully.")

# İndeksleme isteği modeli
class IndexRequest(BaseModel):
    full_document: str
    document_id: str
    metadata: dict
    index_name: str
    max_document_length: int = 4096
    # split_documents: bool

# İndeksleme endpoint'i
@app.route("/index", methods=["POST"])
def index_document():
    try:

        data = request.get_json()
        # Gelen JSON verisini al ve Pydantic ile doğrula
        if not data.get("full_document"):
            raise ValueError("full_document field is required.")

        if not data.get("document_id"):
            raise ValueError("document_id field is required.")

        if not data.get("metadata"):
            raise ValueError("metadata field is required.")

        if not data.get("index_name"):
            raise ValueError("index_name field is required.")


        metadata = data.get("metadata")
        print(f"Metadata length: {len(metadata)}")
        # index_request = IndexRequest(**data)
        full_document = data.get("full_document")
        document_ids = data.get("document_id")
        # document_ids = [tuple(doc_id) if isinstance(doc_id, list) else doc_id for doc_id in data.get("document_id")]

        # Uzunluklarını kontrol et
        print(f"full_document length: {len(full_document)}")
        print(f"document_ids list length: {len(document_ids)}")

        print(f"Collection: {full_document}")
        print(f"Document IDs: {document_ids}")


        for doc in full_document:
            print(f"Document: {doc}, Type: {type(doc)}")

        for doc_id in document_ids:
            print(f"Document ID: {doc_id}, Type: {type(doc_id)}")

        for meta in metadata:
          print(f"Metadata item: {meta}, Type: {type(meta)}")
        # index_request nesnesini dict'e çevir
        # index_request_dict = index_request.dict()
        # print(index_request_dict)
        print("Received JSON data:", data, flush=True)

        # RAG modelini kullanarak indeksle
        # RAG.add_to_index(
        #     new_collection=full_document,
        #     new_document_ids=document_ids,
        #     new_document_metadatas=metadata,
        #     index_name=data.get("index_name"),
        #     max_document_length=data.get("max_document_length"),
        #     #use_faiss=True
        #     split_documents=False,
        # )

        # if os.path.exists(index_path+"/colbert/indexes/"+data.get("index_name")):
        # if os.path.exists(".ragatouille/colbert/indexes/"+data.get("index_name")):
        #     print(f"Model dosyası {index_path} bulundu, indexe ekleniyor...")
        #     # Dosya yolundan modeli yükle
        #     # RAG.add_to_index(
        #     #   new_collection=full_document,
        #     #   new_document_ids=document_ids,
        #     #   new_document_metadatas=metadata,
        #     #   index_name=data.get("index_name"),
        #     # )
        #     RAG.index(
        #       collection=full_document,
        #       document_ids=document_ids,
        #       document_metadatas=metadata,
        #       index_name=data.get("index_name"),
        #       max_document_length=data.get("max_document_length"),
        #       #use_faiss=True
        #       split_documents=False,
        #     )
        # else:
        print(f"Model dosyası bulunamadı, {pretrained_model_name} index oluşturuluyor...")
        RAG.index(
          collection=full_document,
          document_ids=document_ids,
          document_metadatas=metadata,
          index_name=data.get("index_name"),
          max_document_length=data.get("max_document_length"),
          #use_faiss=True
          split_documents=False,
        )

        return data.get("index_name")

    except ValidationError as e:
        # Eğer doğrulama hatası olursa
        return jsonify({"error": e.errors()}), 400


# Sorgu endpoint'i
@app.route("/search", methods=["POST"])
def search_rag():
    try:
        print("searchaaaa")
        # Gelen JSON verisini al ve Pydantic ile doğrula
        data = request.json
        if not data.get("index_name"):
            raise ValueError("index_name field is required.")

        if not data.get("query"):
            raise ValueError("query field is required.")

        # data'yı yazdır
        # print("Received JSON data:", data)

        # query_request'i yazdır
        # print("QueryRequest object:", query_request)

        # Sorguyu böl ve RAG modelinde ara
        queries = data.get("query").split('|')
        print("QueryRequest queries:", queries)
        rag = RAGPretrainedModel.from_index(index_path+"/colbert/indexes/"+data.get("index_name"))

        docs = rag.search(query=queries, index_name=data.get("index_name"))
        print("doc",docs)
        if data.get("rerank"):
            print("rerank")
            docs = rag.rerank(query=queries, documents=[doc['content'] for doc in docs], k=data.get("k") or 5)
            print("rerankink docs",docs)

        return jsonify({"result": docs})

    except ValidationError as e:
        # Eğer doğrulama hatası olursa
        return jsonify({"error": e.errors()}), 400

# Start the Flask server in a new thread
if __name__ == "__main__":
    # threading.Thread(target=app.run, kwargs={"use_reloader": False, "debug": True}).start()
    app.run(host="0.0.0.0")