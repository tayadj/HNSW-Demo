# HNSW Demo

Build via `docker compose up -d --build` and attach to it with `docker attach core`. 

You might want to try different things with it. For example:
```python
>>> engine = Engine()
>>> engine.llm("Hi, who are you?")
'Hello there! 👋 I am an AI Assistant, here to help you with any questions or tasks you might have. How can I assist you today? 😊'
>>> engine.llm("Could you tell me any fact from 2021?")
"Hello there! I'd be happy to share a fact from 2021 for you! 😊\n\nOne exciting thing that happened in 2021 is the launch of the **James Webb Space Telescope** in December. It's designed to observe the most distant events and objects in the universe, like the formation of the first galaxies! ✨"
```