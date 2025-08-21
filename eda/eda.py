from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv

# Load OpenAI API Key from environment variables
load_dotenv()

# Initialize OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class HealthChat:
    def __init__(self, system_prompt: str = "", max_completion_tokens: int = 1000, model: str = "gpt-4"):
        """
        Initializes the HealthChat class to interact with OpenAI GPT-4 via OpenAI API.

        Args:
            system_prompt (str): System-level instructions for the model.
            max_completion_tokens (int): Max token limit for the model's response.
            model (str): The model ID to use with OpenAI API. Default is GPT-4.
        """
        self.system_prompt = system_prompt
        self.max_completion_tokens = max_completion_tokens
        self.model = model

    def ask(self, csv_summary: str, question: str) -> str:
        """
        Asks the GPT-4 model a question based on the provided CSV summary.
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_completion_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"CSV Summary: {csv_summary}\n\nQuestion: {question}"}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error invoking OpenAI API: {e}")
            return f"Error invoking model: {e}"


def generate_graph(csv_file: str, graph_type: str = None, column: str = None) -> list:
    """
    Generates graphs based on the graph type specified.
    If graph_type is None -> generates all types of graphs.
    If column is None -> generates graphs for all applicable columns.

    Args:
        csv_file (str): Path to the CSV file.
        graph_type (str, optional): Type of the graph ('dist', 'time', 'corr', or 'cat').
        column (str, optional): The column to use for the graph, if applicable.

    Returns:
        list: A list of base64 encoded graph images.
    """
    df = pd.read_csv(csv_file)
    graphs = []

    def save_plot(fig):
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format="png")
        image_stream.seek(0)
        img_base64 = base64.b64encode(image_stream.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    # If no graph_type, generate all types
    graph_types = [graph_type] if graph_type else ["dist", "time", "corr", "cat"]

    for gtype in graph_types:
        if gtype == "dist":  # Distribution plots
            numeric_cols = [column] if column else df.select_dtypes(include="number").columns
            for col in numeric_cols:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=20, edgecolor="black")
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                graphs.append(save_plot(fig))

        elif gtype == "time":  # Time series plots
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                numeric_cols = [column] if column else df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    ax.plot(df["Date"], df[col])
                    ax.set_title(f"Time Series of {col}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(col)
                    graphs.append(save_plot(fig))

        elif gtype == "corr":  # Correlation heatmap
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            cax = ax.matshow(corr, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_title("Correlation Heatmap")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)
            graphs.append(save_plot(fig))

        elif gtype == "cat":  # Categorical plots
            categorical_cols = [column] if column else df.select_dtypes(exclude="number").columns
            for col in categorical_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Categorical Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                graphs.append(save_plot(fig))

        else:
            raise ValueError("Invalid graph type. Choose from 'dist', 'time', 'corr', or 'cat'.")

    return graphs


def ask_openai(csv_file: str, question: str) -> str:
    """
    Summarizes the CSV file and asks OpenAI GPT-4 for an answer to the given question.

    Args:
        csv_file (str): Path to the CSV file.
        question (str): The question to ask GPT-4.

    Returns:
        str: The answer from GPT-4.
    """
    # Read the CSV and prepare a summary
    df = pd.read_csv(csv_file)
    data_summary = df.describe(include='all').to_string()

    # Initialize HealthChat and ask OpenAI GPT-4
    health_chat = HealthChat(system_prompt="You are an AI assistant that performs data analysis and visualizations.")
    answer = health_chat.ask(data_summary, question)

    return answer
