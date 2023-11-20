# đây là chương trình dùng để test thử cách chương trình hoạt động
# chương trình sẽ nhận input là 1 file txt bất kì từ user
# sau khi nhận được input chương trình sẽ trích xuất các trường và trả về file 

import sys
import fitz
import hashlib
import json
import re
import os
import torch
import pytesseract
import uuid
from PIL import Image
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from elasticsearch import Elasticsearch
from flask import Flask, request, jsonify, send_file
from PyPDF2 import PdfFileReader
from docx import Document
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Thư mục chứa tệp đầu vào
input_directory = "uploads"
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['EXTRACTED_TEXT_FOLDER'] = "extracted_text"
app.config['JSON_FOLDER'] = "json_output"
data_directory = "/Users/nguyen/Desktop/Docker/system/draf_code/json_output"
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])

# Load fine-tuned classification model
classification_model = BertForSequenceClassification.from_pretrained('/Users/nguyen/Desktop/Docker/BERT/fine_tuned_model/checkpoint-1000', num_labels=5)  


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=1024)
# model = BertModel.from_pretrained('bert-base-uncased', max_position_embeddings=1024, ignore_mismatched_sizes=True)




# Tạo thư mục đầu ra cho JSON nếu nó chưa tồn tại
if not os.path.exists(app.config['JSON_FOLDER']):
    os.mkdir(app.config['JSON_FOLDER'])

id_counter = 1
bulk_actions = []

def convert_txt_to_json(file_path, filename, image, content_hash ):
    global id_counter
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        match = re.search(r'.+?(?=\n\n)', file_content, re.DOTALL)
        if match and image == False:
            title = match.group(0).strip()
            content = file_content[match.end():].strip()
        else:
            title = "image"
            content = file_content

        # content_hash = calculate_content_hash(content)
        content_hash = content_hash
        # if is_document_exists("my_index", content_hash):
        #     print(f"Document with content hash {content_hash} already exists. Skipping.")
        #     sys.exit()
        
        category_label = classify_text(content)
        category_names = ["business", "entertainment", "politic", "sport", "tech"]
        category = category_names[category_label]

############# BERT model  ##############################
# dùng BERT model để tính toán giá trị của semantic từ đó scó thể tìm kiếm dựa trên nội dung
        input_ids = tokenizer.encode(content, add_special_tokens=True, max_length=1024)

        # input_ids = tokenizer.encode(content, add_special_tokens=True)
        with torch.no_grad():
            outputs = model(torch.tensor(input_ids).unsqueeze(0))
            semantic_representation = outputs.last_hidden_state.mean(dim=1).tolist()[0]
   
        id = str(id_counter).zfill(3)
        id_counter += 1
        category = category

        json_data = {
            "filename": filename,
            "title": title,
            "content": content,
            "semantic": semantic_representation,
            "category": category,
            "page" : "1",
            "content_hash": content_hash
        }
        
        json_output_path = os.path.join(app.config['JSON_FOLDER'], os.path.splitext(filename)[0] + ".json")

        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False)

        add_bulk_index_action("my_index", id, title, content, category)

# chuyển đổi từ các định dạng văn bản khác sang txt 
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

# các hàm dùng để ứng dụng trong việc tìm kiếm dựa trên nội dung
def search_by_semantic(user_input):

    semantic_results = []

    # input_ids = tokenizer.encode(user_input, add_special_tokens=True)
    input_ids = tokenizer.encode(user_input, add_special_tokens=True, max_length=1024)

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
            'semantic_score': semantic_score,
            'content_hash': hit['_source']['content_hash']

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

#####kiểm tra xem file mà user upload đã tồn tại chưa######

# calculate_content_hash tính giá trị hash của phân content
def calculate_content_hash(content):
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return content_hash

def is_document_exists(index, content_hash):
    query = {
        "query": {
            "term": {
                "content_hash": content_hash
            }
        }
    }
    result = es.search(index=index, body=query)
    return result["hits"]["total"]["value"] > 0

def add_document_to_elasticsearch(index, filename, content):
    content_hash = calculate_content_hash(content)

    if is_document_exists(index, content_hash):
        print("Tệp đã tồn tại và không cần cập nhật")
        pass
    else:
        es.index(index=index, body={
            "filename": filename,
            "content_hash": content_hash
        })


def search_images(query):
    # Thực hiện truy vấn Elasticsearch để tìm kiếm hình ảnh dựa trên nội dung
    search_body = {
        "query": {
            "match": {
                "content": query
            }
        }
    }
    results = es.search(index="my_index", body=search_body)

    # Trả về danh sách các hình ảnh tương ứng
    image_paths = []
    for hit in results['hits']['hits']:
        filename = os.path.splitext(hit['_id'])[0] + ".jpg"
        image_paths.append(filename)

    # Chuyển danh sách tên tệp hình ảnh thành kết quả tìm kiếm
    image_results = []
    for image_path in image_paths:
        image_entry = {
            'title': 'Image',
            'filename': image_path,
            'highlights':  hit.get('highlight', {}).get('content', []),
            'semantic_score': hit['_score'],
            'category': 'image',
            'content_hash': ''
        }
        image_results.append(image_entry)

    return image_results

#classification_model
def classify_text(text):
    
    inputs = tokenizer(text, return_tensors="pt")
    
    
    with torch.no_grad():
        outputs = classification_model(**inputs)

    
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
    return predicted_label


######## API #########

@app.route('/converttojson', methods=['POST'])
def convert_to_json():
    if not os.path.exists(app.config['JSON_FOLDER']):
        os.mkdir(app.config['JSON_FOLDER'])

    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"})

    image = False

    if uploaded_file:

        # Lưu tệp được tải lên vào thư mục tạm thời
        filename = secure_filename(uploaded_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(temp_file_path)

        # # Thực hiện phân loại văn bản
        with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
            content = txt_file.read()
            category_label = classify_text(content)
            category_names = ["business", "entertainment", "politic", "sport", "tech"]
            category = category_names[category_label]

        #thực hiện chuyển đổi văn bản sang định dạng txt sau đó chuyển sang file json
        if filename.lower().endswith(".docx"):
            docx_text = convert_docx_to_txt(temp_file_path)

            temp_txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + ".txt")
            with open(temp_txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(docx_text)

            with open(temp_txt_file_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                json_data = convert_txt_to_json(temp_txt_file_path,filename, image)

        elif filename.lower().endswith(".pdf"):
            pdf_text = convert_pdf_to_text(temp_file_path)

            temp_txt_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + ".txt")
            with open(temp_txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(pdf_text)

            json_data = convert_txt_to_json(temp_txt_file_path, filename, image)
        
        elif filename.lower().endswith(".txt"):
            with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                content_hash = calculate_content_hash(content)

            if is_document_exists("my_index", content_hash):
                return jsonify({"error": "Tài liệu đã tồn tại trong hệ thống."})
            else:
                json_data = convert_txt_to_json(temp_file_path, filename, image, content_hash )

        elif filename.lower().endswith((".jpg", ".png")):
            # Sử dụng OCR để trích xuất nội dung từ hình ảnh
            image = True
            image = Image.open(temp_file_path)
            ocr_text = pytesseract.image_to_string(image, lang='eng')   
            
            # Tạo tệp văn bản (txt) từ văn bản đã trích xuất
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
            
            with open(temp_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(ocr_text)

            json_data = convert_txt_to_json(temp_file_path, filename, image)

        else:
            return jsonify({"error": "Unsupported file format"})

        json_output_path = os.path.join(app.config['JSON_FOLDER'], os.path.splitext(filename)[0] + ".json")
        with open(json_output_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

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

    # Phân loại văn bản để xác định category
    category_label = classify_text(user_input)
    category_names = ["business", "entertainment", "politic", "sport", "tech"]
    category = category_names[category_label]

    # Chia truy vấn thành các từ
    words = user_input.split()

    # Tạo một truy vấn bool với truy vấn fuzzy cho các từ quan trọng
    should_queries = []
    for word in words:
        fuzzy_query = {
            "match": {
                "content": {
                    "query": word,
                    "fuzziness": "AUTO"  
                }
            }
        }
        should_queries.append(fuzzy_query)
    
    # Thêm điều kiện tìm kiếm theo category
    category_query = {
        "match": {
            "category": category
        }
    }
    should_queries.append(category_query)
    
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
            'semantic_score': hit['_score'],
            'category' : hit['_source']['category'],
            'page' : "1",
            'content_hash': hit['_source']['content_hash']
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
                    "size": 20
                }
            }
        }
    }
    results = es.search(index="suggestions", body=search_body)
    suggestions = results["suggest"]["text-suggest"][0]["options"]

    unique_suggestions = set()  
    unique_results = []

    for suggestion in suggestions:
        suggestion_text = suggestion["_source"]["suggestion"]
        if suggestion_text not in unique_suggestions:
            unique_suggestions.add(suggestion_text)
            unique_results.append(suggestion_text)

    return jsonify(unique_results)

    # return jsonify([suggestion["_source"]["suggestion"] for suggestion in suggestions])


@app.route('/search_image', methods=['POST'])
def search_image():
    # Kiểm tra xem có tệp hình ảnh được tải lên hay không
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"})

    # Lấy tệp hình ảnh từ yêu cầu
    uploaded_file = request.files['image']

    filename = secure_filename(uploaded_file.filename)
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(temp_file_path)

    image = Image.open(temp_file_path)
    # Trích xuất nội dung từ hình ảnh bằng OCR
    ocr_text = pytesseract.image_to_string(image, lang='eng')

    words = ocr_text.split()

    should_queries = []
    for word in words:
        fuzzy_query = {
            "match": {
                "content": {
                    "query": word,
                    "fuzziness": "AUTO"  # Đặt mức độ fuzzy tại "AUTO" hoặc giá trị khác theo mong muốn
                }
            }
        }
        should_queries.append(fuzzy_query)

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
            'filename': hit['_source']['filename'],
            'title': hit['_source']['title'],
            'highlights': hit.get('highlight', {}).get('content', []),
            'semantic_score': hit['_score'],
            'category' : hit['_source']['category'],
            'content_hash': hit['_source']['content_hash']
        }
        response.append(entry)

    # Sắp xếp kết quả theo điểm số semantic giảm dần
    response = sorted(response, key=lambda x: x['semantic_score'], reverse=True)
    
    return jsonify(response)

if __name__ == '__main__':

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    app.run()


