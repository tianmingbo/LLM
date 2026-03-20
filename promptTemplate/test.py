from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Say {foo}")
prompt = prompt.format(foo="bar")

print(prompt)
