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

# first_load = RAGPretrainedModel.from_pretrained(pretrained_model_name,index_root=index_path)

# Kaydetmek istediğin modelin path'i

app = Flask(__name__)
# Update any base URLs to use the public ngrok URL

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 16 MB

print("Model loaded successfully.")

from collections import OrderedDict

class ModelCache:
    def __init__(self, max_size=3):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get_model(self, index_name, index_path):
        if index_name in self.cache:
            self.cache.move_to_end(index_name)
            return self.cache[index_name]
        else:
            # Load a new model and add to cache
            model = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
            self.cache[index_name] = model
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return model

    def update_model(self, index_name):
        self.cache[index_name] = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
        docs = self.cache[index_name].search(query="invalidate", index_name=index_name)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

model_cache = ModelCache(max_size=25)


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
        full_document = data.get("full_document")
        document_ids = data.get("document_id")
        deleted_document_id = data.get("deleted_document_id")
        index_name = data.get("index_name")
        # document_ids = [tuple(doc_id) if isinstance(doc_id, list) else doc_id for doc_id in data.get("document_id")]

        # Uzunluklarını kontrol et
        print(f"full_document length: {len(full_document)}")
        print(f"document_ids list length: {len(document_ids)}")

        
        if os.path.exists(index_path+"/colbert/indexes/"+index_name):
          # RAG = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
          RAG = model_cache.get_model(index_name, index_path)
          if deleted_document_id:
            print(f"Modeldeki {deleted_document_id} ids siliniyor...")
            RAG.delete_from_index(deleted_document_id,index_name)
          print(f"Model dosyası bulundu, {pretrained_model_name} indexe ekleniyor...")
          RAG.add_to_index(
              new_collection=full_document,
              new_document_ids=document_ids,
              new_document_metadatas=metadata,
              index_name=index_name,
            )
        else:
          RAG = RAGPretrainedModel.from_pretrained(pretrained_model_name,index_root=index_path)
          print(f"Model dosyası bulunamadı, {pretrained_model_name} index oluşturuluyor...")
          RAG.index(
            collection=full_document,
            document_ids=document_ids,
            document_metadatas=metadata,
            index_name=index_name,
            max_document_length=data.get("max_document_length"),
            # use_faiss=True,
            split_documents=False,
          )

        # Update the model cache with the latest model after modification
        model_cache.update_model(index_name)
        
        return index_name

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

        index_name = data.get("index_name")

        # data'yı yazdır
        # print("Received JSON data:", data)

        # query_request'i yazdır
        # print("QueryRequest object:", query_request)

        # Sorguyu böl ve RAG modelinde ara
        queries = data.get("query").split('|')
        print("QueryRequest queries:", queries)
        # rag = RAGPretrainedModel.from_index(index_path+"/colbert/indexes/"+index_name)
        rag = model_cache.get_model(data.get("index_name"), index_path)
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

# Delete endpoint'i
@app.route("/delete", methods=["POST"])
def delete_rag():
    try:
        print("delete")
        # Gelen JSON verisini al ve Pydantic ile doğrula
        data = request.json
        if not data.get("deleted_document_id"):
            raise ValueError("deleted_document_id field is required.")
        
        if not data.get("index_name"):
            raise ValueError("index_name field is required.")  
          
        deleted_document_id = data.get("deleted_document_id")
        index_name = data.get("index_name")  
        
        if os.path.exists(index_path+"/colbert/indexes/"+index_name):
          RAG = model_cache.get_model(index_name, index_path)
          
          # RAG = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
            
          RAG.delete_from_index(deleted_document_id,index_name)
          model_cache.update_model(index_name)
          return jsonify({"result": "ok"})

        return jsonify({"result": "false"})

    except ValidationError as e:
        # Eğer doğrulama hatası olursa
        return jsonify({"error": e.errors()}), 400

# Start the Flask server in a new thread
if __name__ == "__main__":
    # threading.Thread(target=app.run, kwargs={"use_reloader": False, "debug": True}).start()
    app.run(host="0.0.0.0")