import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

load_dotenv('.env')
API_KEY:str = os.getenv('API_KEY')

def main():
    """
    Outline:
    1. Create a Temp DB in RAM
    2. Push Pandas DF --> TEMP DB
    3. SQL query

    ### Postgres SQL tables, with their properties:
    #
    # Employee(id, name, department_id)
    # Department(id, name, address)
    # Salary_Payments(id, employee_id, amount, date)
    """

    def create_table_definition_prompt(df):
        """
        This function returns a prompt that informs GPT that we want to work with SQL Tables
        """
        prompt = '''### sqlite SQL table, with its properties:
        #
        # Sales({})
        #
        '''.format(",".join(str(x) for x in df.columns))
        return prompt    

    def prompt_input():
        nlp_text = input("Enter information you want to obtain: ")
        return nlp_text
    
    def combine_prompts(df, query_prompt):
        definition = create_table_definition_prompt(df)
        query_init_string = f"### A query to answer: {query_prompt}\nSELECT"
        return definition+query_init_string        

    def handle_response(response):
        query = response["choices"][0]["text"]
        if query.startswith(" "):
            query = "Select"+ query
        return query


    df = pd.read_csv('sales_data_sample.csv')
    temp_db = create_engine('sqlite:///:memory:', echo=True)
    data = df.to_sql(name='Sales',con=temp_db)

    print(create_table_definition_prompt(df))

    
    
    nlp_text = prompt_input()
    #combine_prompts(df, nlp_text)

    client = OpenAI( api_key = API_KEY)
    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=combine_prompts(df, nlp_text),
    temperature=0,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["#", ";"]
    )

    handle_response(response)

    with temp_db.connect() as conn:
        result = conn.execute(text(handle_response(response)))


    result.all()

    
    with temp_db.connect() as conn:
        result = conn.execute(text("Select ORDERNUMBER, SALES from Sales ORDER BY SALES DESC LIMIT 1"))
        result.all()   

   


if __name__ == "__main__":
    main()