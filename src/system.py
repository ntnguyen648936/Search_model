# đây là chương trình chính thức của hệ thống
# chương trình bao gồm các chức năng search, upload dữ liệu
# cách hoạt động chi tiết có thể đọc ở file readme.txt




from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import os
import docx2txt
from PyPDF2 import PdfFileReader
from bs4 import BeautifulSoup
from docx import Document
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Thư mục chứa tệp đầu vào
input_directory = "test"
# Thư mục đầu ra cho các tệp .txt
output_directory = "output_text_files"
app.config['UPLOAD_FOLDER'] = "output_text_files"
# Kết nối đến Elasticsearch cluster
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}])

# Tạo thư mục đầu ra nếu nó chưa tồn tại
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

def convert_docx_to_txt(file_path):
    text = docx2txt.process(file_path)
    return text

def convert_doc_to_txt(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def convert_pdf_to_txt(file_path):
    text = ""
    pdf = PdfFileReader(open(file_path, "rb"))
    for page_num in range(pdf.getNumPages()):
        page = pdf.getPage(page_num)
        text += page.extractText()
    return text

def convert_html_to_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
        text = soup.get_text()
    return text



@app.route('/converttotext', methods=['POST'])
def convert_to_text():
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

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

        # Tiến hành chuyển đổi tệp thành văn bản
        output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".txt")
        if filename.lower().endswith(".docx"):
            text = convert_docx_to_txt(temp_file_path)
        elif filename.lower().endswith(".doc"):
            text = convert_doc_to_txt(temp_file_path)
        elif filename.lower().endswith(".pdf"):
            text = convert_pdf_to_txt(temp_file_path)
        elif filename.lower().endswith(".html"):
            text = convert_html_to_txt(temp_file_path)
        else:
            return jsonify({"error": "Unsupported file format"})

        # Lưu văn bản đã chuyển đổi thành tệp .txt
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)

        # Xóa tệp tạm thời sau khi đã chuyển đổi
        os.remove(temp_file_path)

    return jsonify({"message": "Conversion completed. Text file saved in the output directory."})

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
        if word.lower() == 'gordon':
            should_queries.append({"fuzzy": {"content": word}})
        else:
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

    results = es.search(index="bbc", body=query, headers=headers)

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

if __name__ == '__main__':
    app.run()
