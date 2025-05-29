import requests
from langchain_community.document_loaders import PyPDFLoader

pdf_url = "https://arxiv.org/pdf/2305.14325.pdf"  # Example: Mistral paper
pdf_filename = "sample.pdf"

response = requests.get(pdf_url)
with open(pdf_filename, "wb") as f:
    f.write(response.content)

def load_pdf_and_show_first_1000_chars(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    full_text = ""
    for doc in documents:
        full_text += doc.page_content + "\n"

    # Print first 1000 characters
    print("\n✅ First 1000 characters of PDF content:\n")
    print(full_text[:1000])

if __name__ == "__main__":
    load_pdf_and_show_first_1000_chars(pdf_filename)
