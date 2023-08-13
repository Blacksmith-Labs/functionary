
@profile
def my_function():
    import uuid
    from typing import List
    from fastapi import FastAPI

    from functionary.openai_types import ChatCompletion, ChatInput, Choice, Function, ChatMessage
    from functionary.inference import generate_message

    app = FastAPI(title="Functionary API")

    MODEL = "musabgultekin/functionary-7b-v1"
    LOADIN8BIT = False

    @profile
    def get_model():
        # this is lazy should be using the modal model class
        import torch
        from transformers import LlamaTokenizer, LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=LOADIN8BIT,
        )
        tokenizer = LlamaTokenizer.from_pretrained(MODEL, use_fast=False)
        return model, tokenizer

    class Model:
        @profile
        def __init__(self):
            model, tokenizer = get_model()
            self.model = model
            self.tokenizer = tokenizer

        @profile
        def generate(
            self,
            messages: List[ChatMessage],
            functions: List[Function],
            temperature: float,
        ):
            return generate_message(
                messages=messages,
                functions=functions,
                temperature=temperature,
                model=self.model,  # type: ignore
                tokenizer=self.tokenizer,
            )

    @profile
    def chat_endpoint(chat_input: ChatInput):
        request_id = str(uuid.uuid4())
        model = Model()

        response_message = model.generate(
            messages=chat_input.messages,
            functions=chat_input.functions,
            temperature=chat_input.temperature,
        )

        return ChatCompletion(
            id=request_id, choices=[Choice.from_message(response_message)]
        )

    chat_input = ChatInput(messages=[ChatMessage(role="assistant")])
    result = chat_endpoint(chat_input)
    print(result.json())


my_function()