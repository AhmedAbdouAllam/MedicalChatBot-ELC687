from Imports import *

def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents
def correct_english_syntax(text,tool):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
def clean_data(text_chunks,tool):
    cleaned_chunks = []
    for chunk in text_chunks:
        chunk.page_content = chunk.page_content.replace("•", " ").replace("\n", " ")
        cleaned_chunks.append(chunk)
    return cleaned_chunks


def process_data(extracted_data,index,tool):
    extracted_data[index].page_content = correct_english_syntax(extracted_data[index].page_content, tool)
    return index, extracted_data[index]
def connect_to_pinecone ():
    pc = Pinecone(api_key=API_KEY)   
    return pc
def delete_index_if_exists_and_create(pc,index_name):
    
    indexes_info = pc.list_indexes()    
    names = [index['name'] for index in indexes_info.indexes]# Check if the index exists
    dimension = 384  # Set the dimension to match your embedding model
    if index_name in names:
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    return indexes_info
def parallel_upsert(index, texts, batch_size=100, max_workers=4,model = None):
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.encode (batch) 
            prepped = [{'id':str(j+(i*batch_size)), 'values':embeddings[j],'metadata':{'text':batch[j]}} for j in range(0,len(embeddings))]
            futures.append(executor.submit(index.upsert, vectors=prepped))
    for future in futures:
        future.result()
def main ():
    extracted_data=load_pdf_file(data='data/')
    print ("loaded")
    tool = language_tool_python.LanguageTool('en-GB')  # You can specify other languages like 'en-GB' for British English
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, extracted_data,i,tool) for i in range(len(extracted_data))]
        results = []
        for future in as_completed(futures):
            index, updated_item = future.result()
            extracted_data[index] = updated_item
    
    text_chunks=text_split(extracted_data)
    cleaned_chunks = clean_data (text_chunks,tool)
    print ("data cleaned")
    pc = connect_to_pinecone()
    index_name = 'medicalindex'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    indexes_info = delete_index_if_exists_and_create(pc,index_name)
    # Connect to the new index
    indexx = pc.Index(index_name)
    model = SentenceTransformer(model_name)
    texts = [t.page_content for t in text_chunks]
    print ("started Upsert")
    parallel_upsert(indexx, texts,model = model)
if (__name__ == "__main__"):
    main ()