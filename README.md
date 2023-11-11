Hướng dẫn chạy chương trình

1. Đầu tiên cần khởi động mở ứng dụng docker và sau đó chạy lệnh sau: (terminal)

docker-compose up -d 

sau đó chạy file system_draf.py


2. tạo 1 index my_index để lưu dữ liệu nếu chưa có: (terminal)

curl -X PUT "http://localhost:9200/my_index"

tiếp theo là tạo 1 index mới là suggestions như sau: (terminal)

curl -X PUT "http://localhost:9200/suggestions" -H "Content-Type: application/json" -d '{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "suggestion": {
        "type": "completion"
      }
    }
  }
}'

curl -X PUT "http://localhost:9200/my_index" -H "Content-Type: application/json" -d '{
    "mappings": {
        "properties": {
            "content_hash": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            }
        }
    }
}'

trong trường hợp bị lỗi có thể thực hiện lệnh sau để xoá hết dữ liệu trong index và thực hiện tạo lại mapping cho index
curl -X DELETE "http://localhost:9200/my_index"


3. Chạy api sau đây với phương thức POST trên postman :  http://127.0.0.1:5000/converttojson

 Api trên sử dụng khi user thêm 1 file vào chương trình thì chương trình 
sẽ chuyển đổi sang file .txt để trích xuất các trường và nhập dữ liệu vào elasticsearch

4. Sau khi đã thêm dữ liệu vào elasticsearch thì có thể dùng api bên dưới với phương thức POST trên postman để search 

http://127.0.0.1:5000/search?q=Maternity



5. dùng API sau để xoá 1 file tuỳ chọn

http://127.0.0.1:5000/deletejson

chọn body -> form-data -> key : "filename" (chọn Text) -> value "002.txt" 


6. dung api sau với phương thức GET trên postman để thực hiện tìm kiếm với suggestion search :

http://127.0.0.1:5000/suggest

để bắt đầu search : chọn params với key : q và value : "input của user"


