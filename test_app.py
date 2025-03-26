from app.main import parse_document

def test_import():
    print("Testing imports...")
    try:
        from app.ocr.document_ocr import DocumentOCR
        from app.preprocessing.text_preprocessor import TextPreprocessor
        from app.structuring.data_structurer import DataStructurer
        print("All imports successful!")
        return True
    except Exception as e:
        print(f"Import error: {str(e)}")
        return False

if __name__ == "__main__":
    test_import() 