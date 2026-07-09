def build_chat_payload(
    messages: list[dict[str, str]],
    uploaded_files: list[str],
    file_payloads: list[dict],
    parsed_documents: list[dict],
    user_query: str,
) -> dict:
    return {
        "messages": messages,
        "user_query": user_query,
        "uploaded_files": uploaded_files,
        "file_payloads": file_payloads,
        "parsed_documents": parsed_documents,
    }
