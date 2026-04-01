import json, logging, os, re
from typing import Any, Dict, List

from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

#import state schema
from backend.src.graph.state import VideoAuditState, ComplainceIssue

#import service
from backend.src.services.video_indexer import VideoIndexerService

#configure the logger
logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)


#Node 1 : INDEXER

def video_indexer_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    This node is reponsible for downloading the video from the provided URL, 
    uploads to Azure video indexer
    extracts the insights
    '''
    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")

    logger.info(f"-----[Node:INDEXER] Processing video: {video_url}")
    local_filename = "temp_audio_video.mp4"

    try:
        vi_service = VideoIndexerService()
        #download the video locally
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Unsupported video source. Only YouTube links are supported!!!")
        azure_video_id = vi_service.upload_video_to_azure(local_path, video_name=video_id_input)
        logger.info(f"Video uploaded to Azure Video Indexer with ID: {azure_video_id}")
        #clean up 
        if os.path.exists(local_path):
            os.remove(local_path)
        
        #wait
        raw_insights = vi_service.wait_for_video_processing(azure_video_id)
        #extract relevant insights
        clean_data = vi_service.extract_data(raw_insights)
        logger.info(f"-----[Node:INDEXER] Extraction complete -----")
        return clean_data
    except Exception as e:
        logger.error(f"Error in video_indexer_node: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": None,
            "ocr_text": []
        }
    
# Node 2 : Compliance Auditor
def audio_content_auditor_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    This node is responsible for RAG to audit the content of the video based on the transcript
    and generate compliance issues if any along with the final report
    '''
    logger.info(f"-----[Node:COMPLIANCE_AUDITOR] querying Knowledge Base & LLM")
    transcript = state.get("trasncript", "")
    if not transcript:
        logger.warning("No transcript available for auditing. Skipping compliance audit.")
        return {
            "final_status": "FAIL",
            "final_report" : "No transcript available for auditing. Compliance audit skipped.",
        }
    #intialize azure clients
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    vectorstore = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        azure_search_index=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )
    # RAG Retrieval
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vectorstore.similarity_search(query_text, k=3)
    retrived_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
                    You are a senior compliance auditor,
                    OFFICIAL COMPLIANCE RULES:
                    {retrived_rules}
                    INSTRUCTIONS:
                    1. Analyze the Transcript and OCR text below
                    2. Identify any compliance issues based on the official rules provided above
                    3. return strictly JSON in the following format:
                    {{
                    "compliance_results": [
                        {{
                        "category": "Category of the compliance issue",
                        "description": "Detailed description of the compliance violation with specific references to the transcript or OCR text",
                        "severity": "Severity level of the issue (Low, Medium, High)",
                        }}
                        ],
                        "status": "FAIL",
                        "final_report": "A detailed markdown report summarizing the compliance issues found, if any. If no issues are found, the report should state that the content is compliant with all rules."
                    }}
                    If there are no compliance issues, set status to PASS and "compliance_results" should be an empty list.
                    """

    user_message = f"""
                    VIDEO_METADATA: {state.get("video_metadata", {})}
                    TRANSCRIPT: {transcript}
                    OCR_TEXT: {ocr_text}
                    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        response_content = response.content
        if "```" in response_content:
            response_content = re.search(r"```json(.*?)```", response_content, re.DOTALL).group(1)
        audit_data = json.loads(response_content.strip())
        return {
            "compliance_issues": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "NO REPORT GENERATED")
        }
    except Exception as e:
        logger.error(f"Error in audio_content_auditor_node: {str(e)}")
        #logging the raw response
        logger.error(f"Raw LLM response: {response_content if 'response_content' in locals() else 'No response content available'}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "final_report": "Error occurred during compliance audit. No report generated."
        }
        
            