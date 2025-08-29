from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import uuid
import time
from typing import Dict, Optional

# Load OpenAI API Key from environment variables
load_dotenv()

# Initialize OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File session management
class FileSessionManager:
    def __init__(self, session_timeout: int = 3600):  # 1 hour timeout
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = session_timeout
        self.upload_dir = "temp_uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def create_session(self, file_content: bytes, filename: str) -> str:
        """Create a new file session and return session ID"""
        session_id = str(uuid.uuid4())
        file_path = os.path.join(self.upload_dir, f"{session_id}_{filename}")
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Store session info
        self.sessions[session_id] = {
            "file_path": file_path,
            "filename": filename,
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session info and update last accessed time"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check if session has expired
        if current_time - session["created_at"] > self.session_timeout:
            self.cleanup_session(session_id)
            return None
        
        # Update last accessed time
        session["last_accessed"] = current_time
        return session
    
    def cleanup_session(self, session_id: str):
        """Remove session and delete associated file"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if os.path.exists(session["file_path"]):
                os.remove(session["file_path"])
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session["created_at"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)

# Global file session manager
file_manager = FileSessionManager()


class HealthChat:
    def __init__(self, system_prompt: str = "", max_completion_tokens: int = 1000, model: str = "gpt-4"):
        self.system_prompt = system_prompt
        self.max_completion_tokens = max_completion_tokens
        self.model = model

    def ask(self, csv_summary: str, question: str) -> str:
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
            return f"Error invoking model: {e}"


def generate_graph_from_session(session_id: str, graph_type: str = None, column: str = None) -> list:
    """Generate graphs using a session ID instead of file path"""
    session = file_manager.get_session(session_id)
    if not session:
        raise ValueError("Invalid or expired session ID")
    
    return generate_graph(session["file_path"], graph_type, column)


def generate_graph(csv_file: str, graph_type: str = None, column: str = None) -> list:
    df = pd.read_csv(csv_file)
    graphs = []

    def save_plot(fig):
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format="png")
        image_stream.seek(0)
        img_base64 = base64.b64encode(image_stream.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    graph_types = [graph_type] if graph_type else ["dist", "time", "corr", "cat"]

    for gtype in graph_types:
        if gtype in ["dist", "time", "cat"]:
            if column is None:
                raise ValueError("Column name is required for this graph type.")
            if column not in df.columns:
                raise ValueError("Your column name is wrong.")

        if gtype == "dist":  # Distribution → only numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("No graph exists for selected column.")
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=20, edgecolor="black")
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            graphs.append(save_plot(fig))

        elif gtype == "time":  # Time series → numeric column + Date col
            if "Date" not in df.columns:
                raise ValueError("No 'Date' column found in CSV for time series.")
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("No graph exists for selected column.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            fig, ax = plt.subplots()
            ax.plot(df["Date"], df[column])
            ax.set_title(f"Time Series of {column}")
            ax.set_xlabel("Date")
            ax.set_ylabel(column)
            graphs.append(save_plot(fig))

        elif gtype == "corr":  # Correlation → always allowed if numeric cols exist
            corr = df.corr(numeric_only=True)
            if corr.empty:
                raise ValueError("No numeric columns available for correlation heatmap.")
            fig, ax = plt.subplots()
            cax = ax.matshow(corr, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_title("Correlation Heatmap")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)
            graphs.append(save_plot(fig))

        elif gtype == "cat":  # Categorical → only non-numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError("No graph exists for selected column.")
            fig, ax = plt.subplots()
            df[column].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Categorical Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            graphs.append(save_plot(fig))

        else:
            raise ValueError("Invalid graph type. Choose from 'dist', 'time', 'corr', or 'cat'.")

    return graphs


def ask_openai_from_session(session_id: str, question: str) -> str:
    """Ask questions using a session ID instead of file path"""
    session = file_manager.get_session(session_id)
    if not session:
        raise ValueError("Invalid or expired session ID")
    
    return ask_openai(session["file_path"], question)


def ask_openai(csv_file: str, question: str) -> str:
    df = pd.read_csv(csv_file)
    data_summary = df.describe(include='all').to_string()

    health_chat = HealthChat(system_prompt="You are an AI assistant that performs data analysis and visualizations.")
    answer = health_chat.ask(data_summary, question)

    return answer


def get_csv_info(session_id: str) -> dict:
    """Get basic information about the CSV file in the session"""
    session = file_manager.get_session(session_id)
    if not session:
        raise ValueError("Invalid or expired session ID")
    
    df = pd.read_csv(session["file_path"])
    
    return {
        "filename": session["filename"],
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "has_date_column": "Date" in df.columns
    }


