from ragatouille import RAGPretrainedModel

import faiss


from flask import Flask, request, jsonify


import os

# import threading
# import json
from dotenv import load_dotenv

load_dotenv()  # .env dosyasını yükler
project_name = os.getenv("COLBERT_PROJECT_NAME", "default-value")


# pretrained model name
pretrained_model_name = os.getenv("MODEL_NAME")
index_path = f"/home/ec2-user/pyton-projects/{project_name}/"


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000  # 16 MB

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
            try:
                model = RAGPretrainedModel.from_index(
                    index_path + "/colbert/indexes/" + index_name
                )
            except Exception as e:
                return jsonify({"result": []})
            self.cache[index_name] = model
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return model

    def update_model(self, index_name):
        try:
            self.cache[index_name] = RAGPretrainedModel.from_index(
                index_path + "/colbert/indexes/" + index_name
            )
        except Exception as e:
            return jsonify({"result": []})
        # Dosyayı açıp JSON verisini yükleyelim
        # with open(index_path + "/colbert/indexes/" + index_name+'/collection.json', 'r', encoding='utf-8') as f:
        #   data = json.load(f)
        #   if not data:
        #       print("collection boş. search index çalışmadı")
        #   else:
        #       print("collection dolu")
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        try:
            docs = self.cache[index_name].search(query=" ", index_name=index_name)

        except Exception as e:
            return jsonify({"result": []})


model_cache = ModelCache(max_size=25)


# İndeksleme endpoint'i
@app.route("/index", methods=["POST"])
def index_document():
    try:

        data = request.get_json()
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

        print(f"full_document length: {len(full_document)}")
        print(f"document_ids list length: {len(document_ids)}")
        print(f"document_ids list: {document_ids}")

        if os.path.exists(index_path + "/colbert/indexes/" + index_name + "/plan.json"):
            # RAG = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
            RAG = model_cache.get_model(index_name, index_path)
            if deleted_document_id:
                print(f"Modeldeki {deleted_document_id} ids siliniyor...")
                RAG.delete_from_index(deleted_document_id, index_name)
            print(f"Model dosyası bulundu, {pretrained_model_name} indexe ekleniyor...")
            RAG.add_to_index(
                new_collection=full_document,
                new_document_ids=document_ids,
                new_document_metadatas=metadata,
                index_name=index_name,
                use_faiss=True,
                split_documents=True,  # Belgeleri otomatik olarak parçalara ayır
            )
        else:
            RAG = RAGPretrainedModel.from_pretrained(
                pretrained_model_name, index_root=index_path
            )
            print(
                f"Model dosyası bulunamadı, {pretrained_model_name} index oluşturuluyor..."
            )
            RAG.index(
                collection=full_document,
                document_ids=document_ids,
                document_metadatas=metadata,
                index_name=index_name,
                # max_document_length=data.get("max_document_length"),
                use_faiss=True,
                split_documents=True,  # Belgeleri otomatik olarak parçalara ayır
                max_document_length=512,  # Her parçanın maksimum token sayısı
            )

        # Update the model cache with the latest model after modification
        model_cache.update_model(index_name)

        return index_name

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Sorgu endpoint'i
@app.route("/search", methods=["POST"])
def search_rag():
    try:
        print("search")
        data = request.json
        if not data.get("index_name"):
            raise ValueError("index_name field is required.")

        if not data.get("query"):
            raise ValueError("query field is required.")

        index_name = data.get("index_name")

        # data'yı yazdır
        # print("Received JSON data:", data)

        # print("QueryRequest object:", query_request)

        # Sorguyu böl ve RAG modelinde ara
        queries = data.get("query").split("|")
        print("QueryRequest queries:", queries)
        rag = model_cache.get_model(index_name, index_path)
        # rag = RAGPretrainedModel.from_index(index_path+"/colbert/indexes/"+index_name)
        try:
            docs = rag.search(query=queries, index_name=index_name)
            print("doc", docs)
            if data.get("rerank"):
                print("rerank")
                docs = rag.rerank(
                    query=queries,
                    documents=[doc["content"] for doc in docs],
                    k=data.get("k") or 5,
                )
                print("rerankink docs", docs)

            return jsonify({"result": docs})
        except Exception as e:
            return jsonify({"result": []})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/delete", methods=["POST"])
def delete_rag():
    try:
        print("delete")
        data = request.json
        if not data.get("deleted_document_id"):
            raise ValueError("deleted_document_id field is required.")

        if not data.get("index_name"):
            raise ValueError("index_name field is required.")

        deleted_document_id = data.get("deleted_document_id")
        index_name = data.get("index_name")
        print("deleted_document_id", deleted_document_id)
        if os.path.exists(index_path + "/colbert/indexes/" + index_name + "/plan.json"):
            RAG = model_cache.get_model(index_name, index_path)

            # RAG = RAGPretrainedModel.from_index(index_path + "/colbert/indexes/" + index_name)
            print(f"Modeldeki {deleted_document_id} ids siliniyor...")
            print(f"Modeldeki {index_name} index_name siliniyor...")
            RAG.delete_from_index(deleted_document_id, index_name)
            model_cache.update_model(index_name)
            return jsonify({"result": "ok"})

        return jsonify({"result": "false"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Start the Flask server in a new thread
if __name__ == "__main__":
    # threading.Thread(target=app.run, kwargs={"use_reloader": False, "debug": True}).start()
    app.run(host="0.0.0.0", port=os.getenv("PORT"))
