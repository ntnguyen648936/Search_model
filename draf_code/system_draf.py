# đây là chương trình dùng để test thử cách chương trình hoạt động
# chương trình sẽ nhận input là 1 file txt bất kì từ user
# sau khi nhận được input chương trình sẽ trích xuất các trường và trả về file 

# ngày 31/10 - chương trình cần cập thêm cách xử lí lấy các trường còn thiếu
import json
import re
import os
import torch
from transformers import BertTokenizer, BertModel
from elasticsearch import Elasticsearch
from flask import Flask, request, jsonify
from PyPDF2 import PdfFileReader
from docx import Document
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Thư mục chứa tệp đầu vào
input_directory = "uploads"

app.config['UPLOAD_FOLDER'] = "uploads"

app.config['JSON_FOLDER'] = "json_output"

data_directory = "/Users/nguyen/Desktop/Docker/system/draf_code/json_output"

es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Tạo thư mục đầu ra cho JSON nếu nó chưa tồn tại
if not os.path.exists(app.config['JSON_FOLDER']):
    os.mkdir(app.config['JSON_FOLDER'])

id_counter = 1
bulk_actions = []



def convert_txt_to_json(file_path, filename):
    global id_counter
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        match = re.search(r'.+?(?=\n\n)', file_content, re.DOTALL)
        if match:
            title = match.group(0).strip()
            content = file_content[match.end():].strip()
        else:
            title = ""
            content = file_content


############# BERT model  ##############################
        input_ids = tokenizer.encode(content, add_special_tokens=True)
        with torch.no_grad():
            outputs = model(torch.tensor(input_ids).unsqueeze(0))
            semantic_representation = outputs.last_hidden_state.mean(dim=1).tolist()[0]
   

        id = str(id_counter).zfill(3)
        id_counter += 1

        category = "business"

        json_data = {
            "filename": filename,
            "title": title,
            "content": content,
            "semantic": semantic_representation,
            "category": category
        }
        json_output_path = os.path.join(app.config['JSON_FOLDER'], os.path.splitext(filename)[0] + ".json")

        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False)

        # Thêm hành động index cho mỗi tệp JSON vào danh sách bulk actions
        add_bulk_index_action("my_index", id, title, content, category)

def convert_docx_to_txt(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def convert_pdf_to_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text    

def add_bulk_index_action(index, id, title, content, category):
    action = {
        "index": {
            "_index": index,
            "_id": id
        }
    }
    source = {
        "filename": id + ".json",
        "title": title,
        "content": content,
        "category": category
    }
    bulk_actions.append(json.dumps(action))
    bulk_actions.append(json.dumps(source))


def search_by_semantic(user_input):

    semantic_results = []

    input_ids = tokenizer.encode(user_input, add_special_tokens=True)
    with torch.no_grad():
        user_input_embedding = model(torch.tensor(input_ids).unsqueeze(0))
        user_input_embedding = user_input_embedding.last_hidden_state.mean(dim=1).tolist()[0]

    search_body = {
        "query": {
            "match_all": {}
        }
    }

    search_results = es.search(index="my_index", body=search_body, size=1000)  # Số lượng tệp JSON bạn muốn tìm kiếm
    for hit in search_results['hits']['hits']:
        filename = hit['_source']['filename']
        semantic_representation = hit['_source']['semantic']

        # Tính toán điểm số semantic sử dụng khoảng cách cosine giữa biểu diễn semantic của truy vấn và biểu diễn semantic của tệp
        semantic_score = cosine_similarity(user_input_embedding, semantic_representation)
        
        related_sentences = extract_related_sentences(hit['_source']['content'], user_input)

        entry = {
            'title': hit['_source']['title'],
            'filename': filename,
            'related_sentences': related_sentences,
            'semantic_score': semantic_score
        }
        semantic_results.append(entry)

    # Sắp xếp kết quả theo điểm số semantic giảm dần
    semantic_results = sorted(semantic_results, key=lambda x: x['semantic_score'], reverse=True)

    return semantic_results

def extract_related_sentences(content, user_input, num_sentences=3):

    sentences = sent_tokenize(content)
    related_sentences = []
    for sentence in sentences:
        if user_input in sentence:
            related_sentences.append(sentence)
        if len(related_sentences) >= num_sentences:
            break
    return related_sentences

def cosine_similarity(embedding1, embedding2):
    # Tính toán độ tương đồng cosine giữa hai biểu diễn semantic
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a ** 2 for a in embedding1) ** 0.5
    magnitude2 = sum(b ** 2 for b in embedding2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Tránh chia cho 0
    return dot_product / (magnitude1 * magnitude2)


# suggestion search
def add_suggestion_to_index(suggestion):
    doc = {"suggestion": suggestion}
    es.index(index="suggestions", body=doc)

def extract_keywords_from_text(text):
    words = re.findall(r'\b\w+\b', text)
    return words



@app.route('/converttojson', methods=['POST'])
def convert_to_json():
    if not os.path.exists(app.config['JSON_FOLDER']):
        os.mkdir(app.config['JSON_FOLDER'])

    # Kiểm tra xem có tệp nào được tải lên không
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    uploaded_file = request.files['file']

    # Kiểm tra xem tệp đã được chọn hay không
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"})

    if uploaded_file:
        # Lưu tệp được tải lên vào thư mục tạm thời
        filename = secure_filename(uploaded_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(temp_file_path)

        # Tiến hành chuyển đổi tệp thành JSON
        if filename.lower().endswith(".docx"):
            docx_text = convert_docx_to_txt(temp_file_path)

            # Tạo một tệp văn bản tạm thời và lưu nội dung DOCX vào đó
            temp_txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + ".txt")
            with open(temp_txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(docx_text)

            # Tiếp tục với việc chuyển đổi tệp văn bản thành JSON
            json_data = convert_txt_to_json(temp_txt_file_path,filename)

        elif filename.lower().endswith(".pdf"):
            # Trích xuất văn bản từ tệp PDF
            pdf_text = convert_pdf_to_text(temp_file_path)

            # Tạo một tệp văn bản tạm thời và lưu nội dung PDF vào đó
            temp_txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + ".txt")
            with open(temp_txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(pdf_text)

            # Tiếp tục với việc chuyển đổi tệp văn bản thành JSON
            json_data = convert_txt_to_json(temp_txt_file_path, filename)
        elif filename.lower().endswith(".txt"):
            json_data = convert_txt_to_json(temp_file_path, filename)
        else:
            return jsonify({"error": "Unsupported file format"})

    

        # Xóa tệp tạm thời sau khi đã chuyển đổi
        json_output_path = os.path.join(app.config['JSON_FOLDER'], os.path.splitext(filename)[0] + ".json")
        with open(json_output_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

            # Gửi dữ liệu JSON lên Elasticsearch
            es.index(index="my_index" ,body=json_data)
        #os.remove(temp_file_path)    


        # Trích xuất từ khóa từ nội dung và thêm vào index gợi ý
        for filename in os.listdir(data_directory):
            if filename.endswith(".json"):
                with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    content = data.get('content', '')
                    keywords = extract_keywords_from_text(content)
                    unique_keywords = list(set(keywords))
                    for keyword in unique_keywords:
                        add_suggestion_to_index(keyword)
                        print(f"Added suggestion: {keyword}")

        return jsonify({"message": "Conversion to JSON and upload to Elasticsearch completed."})



@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.get_json()
        user_input = data.get('q')
    else:
        user_input = request.args.get('q')

    if not user_input:
        return jsonify({"error": "Missing 'q' parameter"})

    # Chia truy vấn thành các từ
    words = user_input.split()

    # Tạo một truy vấn bool với truy vấn fuzzy cho các từ quan trọng
    should_queries = []
    for word in words:
        should_queries.append({"match": {"content": word}})

    bool_query = {"bool": {"should": should_queries}}

    # Sử dụng truy vấn bool trong truy vấn chính
    query = {
        "query": bool_query,
        "highlight": {
            "fields": {
                "content": {}
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    results = es.search(index="my_index", body=query, headers=headers)

    response = []
    for hit in results['hits']['hits']:
        entry = {
            'title': hit['_source']['title'],
            'filename': hit['_source']['filename'],
            'highlights': hit.get('highlight', {}).get('content', []),
            'semantic_score': hit['_score']
        }
        response.append(entry)

    # Sắp xếp kết quả theo điểm số semantic giảm dần
    response = sorted(response, key=lambda x: x['semantic_score'], reverse=True)

    # Nếu không có trùng khớp dựa trên truy vấn ngữ nghĩa, sử dụng tìm kiếm ngữ nghĩa
    if not response:
        # Thực hiện tìm kiếm dựa trên biểu diễn semantic
        semantic_results = search_by_semantic(user_input)
        response = semantic_results

    return jsonify(response)



@app.route('/deletejson', methods=['POST'])
def delete_json():
    # Kiểm tra xem 'filename' đã được gửi từ người dùng hay không
    if 'filename' not in request.form:
        return jsonify({"error": "Missing 'filename' parameter"})

    filename = request.form['filename']
    index = "my_index"  # Thay bằng tên index của bạn

    # Hàm để lấy _id dựa trên filename
    def get_document_id_by_filename(index, filename):
        query = {
            "query": {
                "match": {
                    "filename": filename
                }
            }
        }
        result = es.search(index=index, body=query)
        
        if result["hits"]["total"]["value"] == 0:
            return None
        else:
            return result["hits"]["hits"][0]["_id"]

    # Hàm để xoá tài liệu dựa trên filename
    def delete_document_by_filename(index, filename):
        document_id = get_document_id_by_filename(index, filename)
        if document_id:
            es.delete(index=index, id=document_id)
            return True
        else:
            return False

    # Gọi hàm để xoá tài liệu
    result = delete_document_by_filename(index, filename)
    if result:
        return jsonify({"message": "Dữ liệu đã được xoá thành công."})
    else:
        return jsonify({"error": "Không tìm thấy dữ liệu cần xoá."})


@app.route('/suggest', methods=['GET'])
def get_suggestions():
    user_query = request.args.get('q')
    search_body = {
        "suggest": {
            "text-suggest": {
                "prefix": user_query,
                "completion": {
                    "field": "suggestion",
                    "size": 10
                }
            }
        }
    }
    results = es.search(index="suggestions", body=search_body)
    suggestions = results["suggest"]["text-suggest"][0]["options"]
    return jsonify([suggestion["_source"]["suggestion"] for suggestion in suggestions])




if __name__ == '__main__':
    app.run()
