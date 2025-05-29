import pandas as pd
import decimal
import json
import datetime
from .services.query_generator import generate_sql
from .cors.database import execute_query
from .services.chart_generator import suggest_chart, decimal_to_float
from .services.summary_generator import generate_summary


class NL2SQLProcessor:
    """Main NL2SQL processor class"""

    def __init__(self):
        pass

    def process_query(self, question: str):
        """Process natural language question and return full response"""

        try:
            # Step 1: Generate SQL
            sql_query = generate_sql(question)
            if not sql_query:
                raise Exception("SQL query generation failed.")

            # Step 2: Execute SQL query
            result_data = execute_query(sql_query)

            if not result_data or len(result_data) == 0:
                return {
                    "generated_sql": sql_query,
                    "data": [],
                    "best_chart": None,
                    "selected_columns": {},
                    "summary": "No data available for this query."
                }

            # Step 3: Convert result to serializable format
            result_data = json.loads(json.dumps(result_data, default=decimal_to_float))
            df = pd.DataFrame(result_data)

            # Step 4: Suggest chart
            best_chart, selected_columns, other_settings = suggest_chart(question, df)

            # Step 5: Generate summary
            summary = generate_summary(question, sql_query, df)

            # Step 6: Return final response
            return {
                "generated_sql": sql_query,
                "data": result_data,
                "best_chart": best_chart,
                "selected_columns": selected_columns,
                "summary": summary
            }

        except Exception as e:
            return {
                "generated_sql": None,
                "data": [],
                "best_chart": None,
                "selected_columns": {},
                "summary": f"Error occurred: {str(e)}"
            }


# Create instance for easy import (following company pattern)
nl2sql_processor = NL2SQLProcessor()


def ask_nl2sql(question: str):
    """Standalone function to process NL2SQL query and get response"""

    print(f"Received question: {question}")

    # Use the NL2SQLProcessor class to process query
    return nl2sql_processor.process_query(question)
