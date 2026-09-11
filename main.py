import json
import urllib.request

import hnsw



class Engine:

    EMBEDDING_TAG = "ai/embeddinggemma:300M-Q8_0"
    LLM_TAG = "ai/gemma4:E2B"

    def __init__(self):

        self.index = hnsw.Index()

        documents = [
            "The James Webb Space Telescope, launched in December 2021, is designed to observe the most distant events and objects in the universe, such as the formation of the first galaxies.",
            "In 1911, the Norwegian explorer Roald Amundsen became the first person to reach the geographic South Pole, beating his British rival Robert Falcon Scott by just over a month.",
            "Quantum computers leverage the principles of quantum mechanics, specifically superposition and entanglement, to process complex calculations exponentially faster than classical computers.",
            "The axolotl, a unique species of salamander native to the lake complex of Xochimilco in Mexico City, possesses the remarkable ability to regenerate lost limbs, organs, and even parts of its brain.",
            "Written by Mary Shelley and first published in 1818, 'Frankenstein; or, The Modern Prometheus' is often considered one of the earliest examples of science fiction literature."
        ]

        for document in documents:

            self.index.insert(document, self.embedding(document))

    def embedding(self, content):

        request = urllib.request.Request(
            "http://model-runner.docker.internal/v1/embeddings",
            data = json.dumps({
                "model": self.EMBEDDING_TAG,
                "input": content
            }).encode("utf-8"),
            headers = {
                "Content-Type": "application/json"
            }
        )

        try:

            with urllib.request.urlopen(request) as response:

                result = json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exception:

            print(f"HTTP Error: {exception.code} - {exception.reason}")
            print(exception.read().decode())

        return result["data"][0]["embedding"]

    def llm(self, query):

        request = urllib.request.Request(
            "http://model-runner.docker.internal/v1/chat/completions",
            data = json.dumps({
                "model": self.LLM_TAG,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"You are AI Assistant. Answer using knowledge context, if no relevant information found - "
                            f"tell the user you don't know how to help. Also, keep friendly vibes.\n"
                            f"\n"
                            f"User's query: {query}\n"
                            f"\n"
                            f"Knowledge context:\n"
                            f"{'\n'.join([node.content for node in self.index.search(self.embedding(query), 3)])}"
                        )
                    }
                ]
            }).encode("utf-8"),
            headers = {
                "Content-Type": "application/json"
            }
        )

        try:

            with urllib.request.urlopen(request) as response:

                result = json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exception:

            print(f"HTTP Error: {exception.code} - {exception.reason}")
            print(exception.read().decode())

        return result["choices"][0]["message"]["content"]



if __name__ == "__main__":

    engine = Engine()
