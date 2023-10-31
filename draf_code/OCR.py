from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    try:
        # Đọc hình ảnh từ đường dẫn
        image = Image.open(image_path)

        # Sử dụng pytesseract để trích xuất văn bản từ hình ảnh
        extracted_text = pytesseract.image_to_string(image)

        return extracted_text
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None

if __name__ == "__main__":
    # Nhập đường dẫn hình ảnh từ người dùng
    image_path = input("Nhập đường dẫn hình ảnh: ")

    # Thực hiện OCR để trích xuất văn bản từ hình ảnh
    extracted_text = extract_text_from_image(image_path)

    if extracted_text:
        print("Văn bản trích xuất từ hình ảnh:")
        print(extracted_text)
    else:
        print("Không thể trích xuất văn bản từ hình ảnh.")
