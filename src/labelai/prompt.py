def get_prompt_template(title: str, abstract: str, category_list: list[str]) -> str:
    return f"""  # nosec B608
You are a document classification assistant. Your task is to classify the given document into the most appropriate category from the provided list of categories. The document consists of a title and an abstract. The categories are provided with their IDs and full hierarchical paths and definitions in the following format:

ID: [ID], PATH: Level 1 > Level 2 > ... > Level 6 (definition)

For example:

ID: medtop:20000842, PATH: sport > competition discipline > track and field > relay run (A team of athletes run a relay over identical distances)

Here is the document to classify:

Title: {title}

Abstract: {abstract}

Here are the possible categories:

{category_list}

Select the most appropriate category for this document from the list above. Provide only the ID of the selected category (e.g., medtop:20000842). Do not invent a new category; choose only from the provided options. Do not mention any of the irrelevat options in your output. Your output must only contain the most appropriate category and notthing else.
"""
