#%%
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""
PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

#%%
class RAGBase:
    """
    A helper class for RAG retrieval and answering. 
    """
    def __init__(self, 
                 index, 
                 llm_client, 
                 instructions=INSTRUCTIONS, 
                 prompt_template=PROMPT_TEMPLATE, 
                 course="llm-zoomcamp", 
                 model="gpt-5.4-mini"
                 ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, question, num_results=5):
        """Search for relevant documents in the index based on the question and course.

        Args:
            question (str): The question to search for.
            num_results (int, optional): The number of search results to return. Defaults to 5.
        Returns:
            list: A list of search results.
        """
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}
        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )


    def build_context(self, search_results):
        """Builds a context string from search results.

        Args:
            search_results (list): A list of search results.

        Returns:
            str: A context string.
        """
        lines = []
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        """Builds a prompt string from a question and search results.

        Args:
            question (str): The question to include in the prompt.
            search_results (list): A list of search results.

        Returns:
            str: A prompt string.
        """
        context = self.build_context(search_results)

        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip()

    def llm_with_history(self, instructions, user_prompt, model="gpt-5.4-mini"):
        """Generates a response from the LLM based on instructions and user prompt.

        Args:
            instructions (str): The instructions for the LLM.
            user_prompt (str): The user prompt for the LLM.
            model (str, optional): The model to use. Defaults to "gpt-5.4-mini".

        Returns:
            str: The response from the LLM.
        """
        message_history = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_client.responses.create(
            model=model,
            input=message_history
        )

        return response.output_text

    def llm(self, prompt):
        """Generates a response from the LLM based on a prompt.

        Args:
            prompt (str): The prompt for the LLM.
        Returns:
            str: The response from the LLM.
        """
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response.output_text

    def rag(self, query):
        """
        Performs a RAG (Retrieval-Augmented Generation) operation.

        Args:
            query (str): The query to search for and generate a response.

        Returns:
            str: The generated response.
        """
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

