'''
This defines the DAG(Directed Acyclic Graph) that orchestrates the video compliance audit
process.
It connects the nodes uding the StateGraph from LangGraph and defines the flow of data between 
the nodes.


START -> video_indexer_node -> audio_content_auditor_node -> END
'''

from langgraph.graph import END, StateGraph
from backend.src.graph.state import VideoAuditState
from backend.src.graph.nodes import (
    video_indexer_node,
    audio_content_auditor_node
)

def create_graph():
    '''
    Constructs and compiles the graph using the defined nodes and state schema.
    returns the compiled graph object that can be executed with the initial state.
    '''

    #instiantiate the graph
    workflow = StateGraph(VideoAuditState)
    #add nodes to the graph
    workflow.add_node("Video Indexer Node", video_indexer_node)
    workflow.add_node("Audio Content Auditor Node", audio_content_auditor_node)

    workflow.set_entry_point("Video Indexer Node")

    #define the edges
    workflow.add_edge("Video Indexer Node", "Audio Content Auditor Node")

    workflow.add_edge("Audio Content Auditor Node", END)

    app = workflow.compile()
    return app


app = create_graph()
