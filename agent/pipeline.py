from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.steps import (
    check_input,
    predict_price,
    get_market_data,
    use_fallback,
    find_similar_homes,
    write_report,
    add_disclaimer,
    route_after_retrieval,
)


def _build():
    graph = StateGraph(AgentState)

    graph.add_node("check_input", check_input)
    graph.add_node("predict_price", predict_price)
    graph.add_node("get_market_data", get_market_data)
    graph.add_node("use_fallback", use_fallback)
    graph.add_node("find_similar_homes", find_similar_homes)
    graph.add_node("write_report", write_report)
    graph.add_node("add_disclaimer", add_disclaimer)

    graph.set_entry_point("check_input")
    graph.add_edge("check_input", "predict_price")
    graph.add_edge("predict_price", "get_market_data")
    graph.add_conditional_edges(
        "get_market_data",
        route_after_retrieval,
        {"end": END, "fallback": "use_fallback", "continue": "find_similar_homes"},
    )
    graph.add_edge("use_fallback", "find_similar_homes")
    graph.add_edge("find_similar_homes", "write_report")
    graph.add_edge("write_report", "add_disclaimer")
    graph.add_edge("add_disclaimer", END)

    return graph.compile()


pipeline = _build()
