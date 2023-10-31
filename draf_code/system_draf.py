# đây là chương trình dùng để test thử cách chương trình hoạt động
# chương trình sẽ nhận input là 1 file txt bất kì từ user
# sau khi nhận được input chương trình sẽ trích xuất các trường và trả về file 

# ngày 31/10 - chương trình cần cập thêm cách xử lí lấy các trường còn thiếu

from elasticsearch import Elasticsearch
from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfFileReader
from docx import Document
from werkzeug.utils import secure_filename
import json
import re

app = Flask(__name__)

# Thư mục chứa tệp đầu vào
input_directory = "uploads"

app.config['UPLOAD_FOLDER'] = "uploads"

app.config['JSON_FOLDER'] = "json_output"

es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])


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

        id = str(id_counter).zfill(3)
        id_counter += 1

        category = "business"

        json_data = {
            "filename": filename,
            "title": title,
            "content": content,
            "category": category
        }
        #json_output_path = os.path.join(app.config['JSON_FOLDER'], id + ".json")
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

# def save_json_file(json_data, filename):
#     with open(os.path.join(app.config['JSON_FOLDER'], filename), 'w', encoding='utf-8') as json_file:
#         json.dump(json_data, json_file, ensure_ascii=False, indent=4)


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

        # Lưu dữ liệu JSON vào tệp trong thư mục json_output
        # json_filename = os.path.splitext(filename)[0] + ".json"
        # save_json_file(json_data, json_filename)

        # Xóa tệp tạm thời sau khi đã chuyển đổi
        json_output_path = os.path.join(app.config['JSON_FOLDER'], os.path.splitext(filename)[0] + ".json")
        with open(json_output_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

            # Gửi dữ liệu JSON lên Elasticsearch
            es.index(index="my_index" ,body=json_data)
        #os.remove(temp_file_path)        

    #print(jsonify({"message": "Conversion to JSON completed. JSON files saved in the JSON output directory."}))

    # if uploaded_json_file:
    #     json_data = json.load(json_output)

    #     # Gửi dữ liệu JSON lên Elasticsearch
    #     es.index(index="my_index", body=json_data)

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
            'score': hit['_score']
        }
        response.append(entry)

    # Sắp xếp kết quả theo mức độ đúng (score) giảm dần
    response = sorted(response, key=lambda x: x['score'], reverse=True)

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



if __name__ == '__main__':
    app.run()
