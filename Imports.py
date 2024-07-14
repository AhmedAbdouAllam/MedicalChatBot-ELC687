import os
import sys
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as Pinecone_vector
from pinecone import Pinecone

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import language_tool_python
import spacy
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForCausalLM

#from gingerit.gingerit import GingerIt



