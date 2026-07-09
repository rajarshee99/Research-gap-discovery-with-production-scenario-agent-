def normalize_uploaded_files(files: list) -> list[str]:
    return [file.name for file in files]


def serialize_uploaded_files(files: list) -> list[dict]:
    return [
        {
            "name": file.name,
            "content": file.getvalue(),
        }
        for file in files
    ]
