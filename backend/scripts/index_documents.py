import os, glob, logging
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("indexer")

def index_documents():
    '''
    Reads the PDFs, chunks them, and upload them to azure ai search
    '''

    cureent_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(cureent_dir, "../../backend/data")

    logger.info("="*60)
    logger.info("Environment configuration check:")
    logger.info(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    logger.info(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    logger.info(f"Embedding deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')}")
    logger.info(f"AZURE_SEARCH_ENDPOINT: {os.getenv('AZURE_SEARCH_ENDPOINT')}")
    logger.info(f"AZURE_SEARCH_INDEX_NAME: {os.getenv('AZURE_SEARCH_INDEX_NAME')}")
    logger.info("="*60)

    #validate environment variables
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set the missing environment variables and try again.")
        return
    

    # Initialize Azure OpenAI Embeddings
    try:
        logger.info("Initializing Azure OpenAI Embeddings...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        logger.info("Azure OpenAI Embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI Embeddings: {e}")
        logger.error("Please check your Azure OpenAI configuration and try again.")
        return
    
    # Initialize Azure Search Vector Store
    try:
        logger.info("Initializing Azure Search Vector Store...")
        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            azure_search_index=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query,
        )
        logger.info("Azure Search Vector Store initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Azure Search Vector Store: {e}")
        logger.error("Please check your Azure Search configuration and try again.")
        return
    
    # Process each PDF in the data folder
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_folder}. Please add PDF documents to index.")
    logger.info(f"Found {len(pdf_files)} PDF files to process : {[os.path.basename(pdf) for pdf in pdf_files]}")

    all_splits = []

    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing document: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            raw_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(raw_documents)
            for split in splits:
                split.metadata["source"] = os.path.basename(pdf_path)
            all_splits.extend(splits)
            logger.info(f"Document {os.path.basename(pdf_path)} processed successfully with {len(splits)} chunks.")
        except Exception as e:
            logger.error(f"Error processing document {os.path.basename(pdf_path)}: {e}")

    # Upload chunks to Azure Search
    if all_splits:
        logger.info(f"Uploading {len(all_splits)} document chunks to Azure AI Search INDEX: '{os.getenv('AZURE_SEARCH_INDEX_NAME')}'...")
        try:
            vector_store.add_documents(all_splits)
            logger.info("Document chunks uploaded successfully.")
            logger.info("Indexing process completed,Knowledge base is ready for RAG operations.")
            logger.info(f"Total documents indexed: {len(all_splits)}")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Error uploading document chunks to Azure Search: {e}")
            logger.error("Please check your Azure Search configuration and try again.")
    else:
        logger.warning("No document chunks to upload. Please check the PDF processing step for errors.")    

if __name__ == "__main__":
    index_documents()

