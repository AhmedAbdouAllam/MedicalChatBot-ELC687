from Imports import *


def connect_to_pinecone ():
    pc = Pinecone(api_key=API_KEY)   
    return pc

def Intializer ():
    pc = connect_to_pinecone()
    index_name = 'medicalindex'
    indexx = pc.Index(index_name)
    #tokenizer = AutoTokenizer.from_pretrained('gpt2')
    #model = AutoModelForCausalLM.from_pretrained('gpt2')
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    modelx = SentenceTransformer(model_name)

    def embedding_fn(text):
        return modelx.encode(text).tolist()
    
    llm=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama", config={'max_new_tokens': 600,
                              'temperature': 0.7,
                              'context_length': 1500})
    vectordb = Pinecone_vector(index=indexx, embedding = embedding_fn, text_key="text")
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    question = "what is the diseases that have fever?"

    retriever_config = {
        'k': 1  # Adjust k value as per your requirement
    }
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(config = retriever_config),
                                       return_source_documents=False,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} )
    return qa_chain
def main ():

    question = "what are the symptoms of embolus?"
    qa_chain = Intializer ()

    while (True):

        user_input = input("Enter Your question: ")
        result = qa_chain({"query": user_input})
        print("The answer is :" + result["result"])
if (__name__ == "__main__"):
    main ()