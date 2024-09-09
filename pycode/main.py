from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"],
)

code_chain = code_prompt | llm | StrOutputParser()

test_chain = test_prompt | llm | StrOutputParser()

# Invoke the code_chain to get the generated code
code_result = code_chain.invoke({"language": args.language, "task": args.task})
print("Generated Code:", code_result)

# Use the generated code as input to the test_chain
test_result = test_chain.invoke({"language": args.language, "code": code_result})
print("\nGenerated Test:", test_result)