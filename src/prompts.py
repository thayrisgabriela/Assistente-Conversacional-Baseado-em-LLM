'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

qa_template = """
Baseando-se no seguinte contexto, responda detalhadamente à pergunta fornecida. Forneça explicações completas e relevantes:

Contexto: {context}

Pergunta: {question}

Resposta detalhada:
"""
