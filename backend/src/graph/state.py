import operator
from typing import Annotated, List, Dict, Any, Optional, TypedDict

#Error report 
class ComplianceIssue(TypedDict):
    category: str
    description: str #specific detail of the violation
    severity: str
    timestamp: Optional[str]

class VideoAuditState(TypedDict):
    '''
    Defines the data schema for langgraph execution content. This is the state that will be passed between nodes in the graph.
    '''
    #input fields
    video_url: str
    video_id: str

    #ingestion and extraction related fields
    local_file_path: Optional[str] #temporary local path where the video
    video_metadata: Dict[str, Any] 
    transcript: Optional[str] #full transcript of the video
    ocr_text: List[str]

    #analysis output fields
    #list of all the violations found by the system with details
    compliance_issues: Annotated[List[ComplianceIssue], operator.add]

    #final deliverables
    final_status: str # PASS | FAIL
    final_report: str # markdown report

    #system observability fields
    #errors: API timeoouts, system errors, etc
    errors: Annotated[List[str], operator.add]
